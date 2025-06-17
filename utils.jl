using BSON, CUDA, Dates, FFTW, Flux, JLD2, Plots, Printf, Statistics

function conv_fft(f::Vector, g::Vector; dx::Number=0.01, rfftP=plan_rfft(f))
    rfftP \ ((rfftP * f) .* (rfftP * g)) * dx
end

struct Model{T1 <: Chain, T2 <: Dense, T3 <: Dense}
    c1::T1
    μ::T2
    T::T3
    c1_with_T
    trainable
end

function Model(ρ_window_size, n_sims; μ_init=zeros(n_sims), T_init=ones(n_sims), c1_with_T=false, trainable=(:c1, :μ), hidden_nodes=[128, 64, 32])
    c1 = Chain(
        Parallel(c1_with_T ? vcat : (ρ_windows, T) -> ρ_windows, identity),
        Dense(ρ_window_size + c1_with_T => hidden_nodes[begin], softplus),
        [Dense(hidden_nodes[i] => hidden_nodes[i+1], softplus) for i in 1:length(hidden_nodes)-1]...,
        Dense(hidden_nodes[end] => 1)
    )
    μ = Dense(reshape(μ_init, 1, :), false, identity)
    T = Dense(reshape(T_init, 1, :), false, identity)
    if :T in trainable && !c1_with_T
        error("You should probably set c1_with_T=true if T is trainable")
    end
    Model(c1, μ, T, c1_with_T, trainable)
end

Flux.@layer Model

Flux.trainable(m::Model) = (; (field => getproperty(m, field) for field in m.trainable)...)

function (m::Model)(ρ_windows, sim_onehots)
    T = m.T(sim_onehots)
    μ = m.μ(sim_onehots)
    c1 = m.c1(ρ_windows, T)
    c1, μ, T
end

function get_weights_Percus(xs)
    xs = xs .- xs[begin] # undo shift of dx/2
    dx = xs[2]
    L = xs[end] + dx
    R = 0.5
    @assert L >= 2*R "To construct the Percus weight functions, the system must have a length of at least 2R."
    @assert count(xs .≈ R) == 1 "The grid is not suitable for the construction of the Percus weight functions. R must be a multiple of dx."
    ω0 = zero(xs)
    ω0[xs.≈R] .= 0.5 / dx
    ω0[2:end] += reverse(ω0[2:end])
    ω1 = zero(xs)
    ω1[xs.<R] .= 1
    ω1[xs.≈R] .= 0.5
    ω1[2:end] += reverse(ω1[2:end])
    ω0, ω1
end

function get_c1_Percus(xs)
    ω0, ω1 = get_weights_Percus(xs)
    conv(f, g) = conv_fft(f, g; dx=xs[2] - xs[1], rfftP=plan_rfft(xs))
    function (ρ, T)
        n0, n1 = conv(ω0, ρ), conv(ω1, ρ)
        ∂ϕ_∂n0 = -log.(1 .- n1)
        ∂ϕ_∂n1 = n0 ./ (1 .- n1)
        -(conv(ω0, ∂ϕ_∂n0) + conv(ω1, ∂ϕ_∂n1))
    end
end

function minimize(L::Number, T::Number, Vext, get_c1::Function; floattype::Type=Float32, dx::Number=0.01, ρ_init=x->0.5, kwargs...)
    xs = collect(floattype, dx/2:dx:L)
    c1 = get_c1(xs)
    if typeof(Vext) <: Function
        Vext = floattype.(Vext.(xs))
    end
    if typeof(ρ_init) <: Function
        ρ_init = floattype.(ρ_init.(xs))
    end
    minimize(xs, T, Vext, c1; floattype, ρ_init, kwargs...)
end

function minimize(xs::AbstractArray, T::Number, Vext::AbstractArray, c1::Function; floattype::Type=Float32, μ::Number=NaN, ρ̄::Number=NaN, α::Number=0.05, ρ_init::AbstractArray=fill(floattype(0.5), size(xs)), maxiter::Int=10000, tol::Number=max(eps(floattype(1e3)), 1e-8), plot_every::Number=0, symmetrize_c1::Bool=false)
    if (isnan(μ) && isnan(ρ̄)) || (!isnan(μ) && !isnan(ρ̄))
        error("Specify either the chemical potential μ or the mean density ρ̄")
    end
    T, μ, ρ̄ = floattype.((T, μ, ρ̄))
    infiniteVext = isinf.(Vext) .|| isnan.(Vext)
    ρ = copy(ρ_init)
    ρEL = copy(ρ)
    i = 0
    while true
        c1_profile = c1(ρ, T)
        if symmetrize_c1
            c1_profile .*= 0.5
            c1_profile .+= 0.5 .* c1(reverse(ρ), T)
        end
        ρEL .= exp.(-Vext ./ T .+ c1_profile)
        if !isnan(μ)
            ρEL .*= exp(μ / T)
        end
        if !isnan(ρ̄)
            ρEL .*= ρ̄ / mean(ρEL)
        end
        ρ .= (1 - α) .* ρ .+ α .* ρEL
        ρ[infiniteVext] .= 0
        clamp!(ρ, 0, Inf)
        Δρmax = maximum(abs, ρ - ρEL)
        i += 1
        if plot_every > 0 && i % plot_every == 0
            display(plot(xs, ρ))
        end
        if Δρmax < tol
            println("Converged (step: $(i), ‖Δρ‖: $(Δρmax) < $(tol))")
            break
        end
        if !isfinite(Δρmax) || i >= maxiter
            error("Did not converge (step: $(i) of $(maxiter), ‖Δρ‖: $(Δρmax), tolerance: $(tol))")
        end
    end
    xs, ρ
end

function read_sim_data(dir; n_max=Inf)
    ρ_profiles = Vector{Vector{Float64}}()
    Vext_profiles = Vector{Vector{Float64}}()
    μ_values = Vector{Union{Missing,Float64}}()
    T_values = Vector{Union{Missing,Float64}}()
    for sim in readdir(dir, join=true)
        file = jldopen(sim, "r")
        xs, ρ, Vext = file["xs"], file["ρ"], file["Vext"]
        μ = "μ" in keys(file) ? file["μ"] : missing
        T = "T" in keys(file) ? file["T"] : missing
        push!(ρ_profiles, ρ)
        push!(Vext_profiles, Vext)
        push!(μ_values, μ)
        push!(T_values, T)
        if length(T_values) >= n_max
            break
        end
    end
    ρ_profiles, Vext_profiles, μ_values, T_values
end

function generate_windows(ρ; window_bins=201)
    if iseven(window_bins)
        error("window_bins should be odd")
    end
    ρ_windows = zeros(eltype(ρ), window_bins, length(ρ))
    pad = window_bins ÷ 2 - 1
    ρpad = vcat(ρ[end-pad:end], ρ, ρ[begin:begin+pad])
    for i in 1:length(ρ)
        ρ_windows[:,i] = ρpad[i:i+window_bins-1]
    end
    ρ_windows
end

function generate_inout(ρ_profiles, Vext_profiles; window_bins=201)
    ρ_windows_all = Vector{Vector{Float32}}()
    sim_onehot_all = Vector{Vector{Float32}}()
    ρ_values_all = Vector{Float32}()
    Vext_values_all = Vector{Float32}()
    for (sim_id, (ρ, Vext)) in enumerate(zip(ρ_profiles, Vext_profiles))
        ρ_windows = generate_windows(ρ; window_bins)
        out = log.(ρ) .+ Vext
        for i in 1:length(ρ)
            if ρ[i] < 0.0001
                continue
            end
            push!(ρ_windows_all, ρ_windows[:,i])
            push!(sim_onehot_all, Float32.(Flux.onehot(sim_id, 1:length(ρ_profiles))))
            push!(ρ_values_all, ρ[i])
            push!(Vext_values_all, Vext[i])
        end
    end
    reduce(hcat, ρ_windows_all), reduce(hcat, sim_onehot_all), ρ_values_all', Vext_values_all'
end

function get_c1_rho_window_size(c1_model)
    input = length(c1_model.layers[2].weight[1,:])
    iseven(input) ? input - 1 : input
end

function get_c1_neural(c1_model, xs; offset=true)
    window_bins = get_c1_rho_window_size(c1_model)
    c1_model = c1_model |> gpu
    function (ρ, T)
        ρ_windows = generate_windows(ρ; window_bins)
        T = fill(T, (1, size(ρ_windows)[2]))
        c1 = c1_model((ρ_windows, T) |> gpu)
        if !offset
            return c1 |> cpu |> vec
        end
        c1_offset = c1_model((zero(ρ_windows), T) |> gpu)
        (c1 - c1_offset) |> cpu |> vec
    end
end

function get_model_input(model_state)
    window_bins = size(model_state[:c1].layers[2].weight)[2]
    n_sims = size(model_state[:μ].weight)[2]
    c1_with_T = model_state[:c1_with_T]
    if c1_with_T
        window_bins -= 1
    end
    window_bins, n_sims
end

function get_hidden_nodes(model_state)
    hidden_nodes = []
    for layer in model_state[:c1].layers[2:end-1]
        push!(hidden_nodes, size(layer.weight)[1])
    end
    hidden_nodes
end

function load_trained_model(model_state_history_savefile)
    model_state_history = JLD2.load(model_state_history_savefile, "model_state_history")
    model_state_last = model_state_history[end]
    window_bins, n_sims = get_model_input(model_state_last)
    model = Model(window_bins, n_sims; c1_with_T=model_state_last[:c1_with_T], trainable=model_state_last[:trainable], hidden_nodes=get_hidden_nodes(model_state_last))
    Flux.loadmodel!(model, model_state_last)
    model
end