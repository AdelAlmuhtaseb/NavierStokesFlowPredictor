import airfrans as af

af.dataset.download(root="data", file_name="Dataset", unzip=True, OpenFOAM=False)
print("done")