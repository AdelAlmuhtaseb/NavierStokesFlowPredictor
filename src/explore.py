import airfrans as af
import matplotlib.pyplot as plt

data, names = af.dataset.load(root="data/Dataset", task="full", train=True)
print("simulations:", len(data))
print("first one:", names[0], data[0].shape, data[0].dtype)

# Quick look at one mesh. I believe the first two columns are x, y positions;
# confirm in the docs (airfrans.readthedocs.io) before relying on this.
s = data[0]
plt.figure(figsize=(8, 4))
plt.scatter(s[:, 0], s[:, 1], s=0.2)
plt.axis("equal")
plt.savefig("results/mesh_example.png", dpi=150)