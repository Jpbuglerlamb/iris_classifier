from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import graphviz

iris = load_iris()
x, y=iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

clif = DecisionTreeClassifier(max_depth=3, random_state=42)
clif.fit(x_train, y_train)

export_graphviz(clif,
                out_file="tree.dot",
                feature_names=iris.feature_names,
                class_names=iris.target_names,
                filled=True, rounded=True,
                special_characters=True)

plt.figure(figsize=(12, 8))
plot_tree(clif, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.savefig("iris_tree_matplotlib.png", dpi=150)
print("Matplotlib PNG saved succesfully")
