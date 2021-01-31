import python.monkdata as m
import python.dtree as dec
import python.drawtree_qt5 as draw
import random
import matplotlib.pyplot as plt


def pick_best_tree(old_tree, validation_set):
    list_of_trees = dec.allPruned(old_tree)
    old_accuracy = dec.check(old_tree, validation_set)
    validation_accuracy = 0
    result = None
    for i in range(len(list_of_trees)):
        temp = list_of_trees[i]
        temp_accuracy = dec.check(temp, validation_set)
        if temp_accuracy > validation_accuracy:
            validation_accuracy = temp_accuracy
            result = temp
    if validation_accuracy < old_accuracy:
        return old_tree
    else:
        pick_best_tree(result, validation_set)
        return result


def main():
    monk1 = m.monk1
    monk2 = m.monk2
    monk3 = m.monk3
    entropy = dec.entropy(m.monk1)
    print("MONK 1: ", entropy)
    entropy = dec.entropy(m.monk2)
    print("MONK 2: ", entropy)
    entropy = dec.entropy(m.monk3)
    print("MONK 3: ", entropy)

    print(dec.bestAttribute(monk1, m.attributes))
    print(("MONK1:"))
    for i in range(6):
        print("Information gain of a" + str(i+1) + " is " + str(dec.averageGain(monk1, m.attributes[i])))
    print(("MONK2:"))
    for i in range(6):
        print("Information gain of a" + str(i+1) + " is " + str(dec.averageGain(monk2, m.attributes[i])))
    print(("MONK3:"))
    for i in range(6):
        print("Information gain of a" + str(i+1) + " is " + str(dec.averageGain(monk3, m.attributes[i])))

    print(("MONK1:"))
    tree = dec.buildTree(monk1, m.attributes)
    print("Training Error: ",1-dec.check(tree, m.monk1))
    print("Test Error: ",1-dec.check(tree, m.monk1test))
    print(("MONK2:"))
    tree = dec.buildTree(monk2, m.attributes)
    print("Training Error: ",1-dec.check(tree, m.monk2))
    print("Test Error: ",1-dec.check(tree, m.monk2test))
    print(("MONK3:"))
    tree = dec.buildTree(monk3, m.attributes)
    print("Training Error: ",1-dec.check(tree, m.monk3))
    print("Test Error: ",1-dec.check(tree, m.monk3test))

    def partition(data, fraction):
        ldata = list(data)
        random.shuffle(ldata)
        breakPoint = int(len(ldata) * fraction)
        return ldata[:breakPoint], ldata[breakPoint:]
    values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    best_test_error = 0
    model_score = []
    best_fraction = 0
    for fraction in values:
        total_test_error = 0
        for j in range(1000):
            monk3train, monk3val = partition(m.monk3, fraction)
            tree = dec.buildTree(monk3train, m.attributes)
            result = pick_best_tree(tree, monk3val)
            total_test_error += dec.check(result, m.monk3test)
        avg_test_error = total_test_error/1000
        model_score.append(avg_test_error)
        if avg_test_error > best_test_error:
            best_test_error = avg_test_error
            best_fraction = fraction
    plt.scatter(values, model_score)
    plt.xlabel("Split fraction")
    plt.ylabel("Test accuracy")
    plt.savefig("Monk3.png")
    plt.show()

    best_test_error = 0
    model_score = []
    best_fraction = 0
    for fraction in values:
        total_test_error = 0
        for j in range(1000):
            monk1train, monk1val = partition(m.monk1, fraction)
            tree = dec.buildTree(monk1train, m.attributes)
            result = pick_best_tree(tree, monk1val)
            total_test_error += dec.check(result, m.monk1test)
        avg_test_error = total_test_error/1000
        model_score.append(avg_test_error)
        if avg_test_error > best_test_error:
            best_test_error = avg_test_error
            best_fraction = fraction
    plt.scatter(values, model_score)
    plt.xlabel("Split fraction")
    plt.ylabel("Test accuracy")
    plt.savefig("Monk1.png")
    plt.show()


    print(best_fraction)
    # draw.drawTree(dec.buildTree(monk1, m.attributes))
    # draw.drawTree(tree)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


