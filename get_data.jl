using Downloads, Tar
Downloads.download("https://www.staff.uni-bayreuth.de/~bt306964/data/neural-mu/data.tar", "data.tar")
Downloads.download("https://www.staff.uni-bayreuth.de/~bt306964/data/neural-mu/models.tar", "models.tar")
Downloads.download("https://www.staff.uni-bayreuth.de/~bt306964/data/neural-mu/predictions.tar", "predictions.tar")
Tar.extract("data.tar", "data")
Tar.extract("models.tar", "models")
Tar.extract("predictions.tar", "predictions")
