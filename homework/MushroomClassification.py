from algorithms.KNN import KNN
from algorithms.Naive_Bayes import NaiveBayes
from algorithms.Perceptron import Perceptron
import random
import time

parse_table = [
   
    ['e', 'p'], #1 classes  
    ['b','c','x','f','k','s'],#2 cap-shape
    ['f', 'g', 'y', 's'], #3 -cap-surface
    ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'], #4 cap-color
    ['t', 'f'], #5 bruise?
    ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'], #6 odor
    ['a', 'd', 'f', 'n'], #7 gill-attachment
    ['c','w','d'], #8 gill=spacing
    ['b', 'n'], #9 gill-size
    ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w','y'],#10 gill-color
    ['e','t'], #11 stalk-shape
    ['b','c','u','e','z','r','?'], #12 stalk-root
    ['f','y','k','s'], #13 stalk-surface-above-ring
    ['f','y','k','s'], #14 stalk-surface-below-ring
    ['n','b','c','g','o','p','e','w','y'], #15 stalk-color-above-ring
    ['n','b','c','g','o','p','e','w','y'], #16 stalk-color-below-ring
    ['p','u'], #17 veil-type
    ['n','o','w','y'], #18 veil-color
    ['n','o','t'], #19 ring-number
    ['c','e','f','l','n','p','s','z'], #20 ring-type
    ['k','n','b','h','r','o','u','w','y'], #21 spore-print-color
    ['a','c','n','s','v','y'], #22 population
    ['g','l','m','p','u','w','d'] #23 habitat


        ]


def read_file(file_path):
    data_points = []
    with open(file_path, 'r') as file:
        for line in file:
            # Strip newline and remove commas, then store each character separately
            data_points.append([char for char in line.strip() if char != ','])  
    return data_points

def parse_to_number(data, table):
    number_array = []
    zipped_array = zip(*data)

    print("table_length" +  str(len(table)))
    zipped_array = tuple(zipped_array)
    print("tuple_length" +  str(len(zipped_array)))
    for i in range(len(table)):
        current_category = table[i] #the attribute class, for example: cap-shape
        print(current_category)
        print("current index =" + str(i))
        current_data_column = list(zipped_array[i])
        print(len(current_data_column))

  #all the data that belongs to that attribute
        for j in range(len(current_category)):# the character that represent 1 attribute
            for k in range(len(current_data_column)):
                if current_category[j] == current_data_column[k]:
                    current_data_column[k] = j
        number_array.append(current_data_column)

    #debug, showing that it has parsed everything
    number_array = zip(*number_array)
    for x in tuple(number_array):
        print(x)

    number_array = list(number_array)
    return number_array




def calculate_accuracy(predictions, true_labels):
    correct = sum([1 for pred, true in zip(predictions, true_labels) if pred == true])
    accuracy = correct / len(true_labels)
    return accuracy

def separate_features_labels(data):
        labels = [point[0] for point in data]
        features = [point[1:] for point in data]
        return features, labels

def split_data(data, train_ratio=0.8):
    # random.shuffle(data)  # Shuffle the data to ensure randomness
    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

def main():
    '''
    while True:
        input_file = input("Please enter the training data file name: ").strip()
        try:
            train_data = read_file(input_file)
            break
        except FileNotFoundError:
            print("Invalid file name! Please enter the training data file name.")
    while True:
        input_file = input("Please enter the test data file name: ").strip()
        try:
            test_data = read_file(input_file)
            break
        except FileNotFoundError:
            print("Invalid file name! Please enter the test data file name.")
    '''

    while True:
        file_choice = input("Enter 1 to load a single file.\nEnter 2 to load separate training and test files.\n").strip()
        
        if file_choice == "1":
            # file_name = input("Please enter the file name: ").strip()
            data = read_file('MushroomData_8000.txt')
            train_data, test_data = split_data(data)
            break
        if file_choice == "2":
            # train_data = input("Please enter the training data file name: ").strip()
            # test_data = input("Please enter the test data file name: ").strip()
            train_data = read_file('MushroomData_8000.txt') # Delete later !!!
            test_data = read_file('MushroomData_Unknwon_100.txt') # Delete later !!!
            train_data = read_file(train_data)
            test_data = read_file(test_data)
        else:
            print("Invalid input! Please enter 1 or 2.")

    train_x, train_y = separate_features_labels(train_data)
    test_x, test_y = separate_features_labels(test_data)

    while True:
        # Ask the user which knapsack algorithm they want to run
        algorithm_type = input("Enter 1 for KNN.\nEnter 2 for Naïve Bayes Classifier.\nEnter 3 for Perceptron.\n").strip()
        
        if algorithm_type == "1":
            knn = KNN(k=20)
            train_x = parse_to_number(train_x, parse_table[1:])
            train_y = parse_to_number(train_y, [parse_table[0]]) # need the extra bracket around the parse_table[0]
            print(train_x)
            print(train_y)
            knn.fit(train_x, train_y)
            start_time = time.time()
            knn_predictions = knn.predict(test_x)
            end_time = time.time()
            accuracy = calculate_accuracy(knn_predictions, test_y)
            print(f"KNN Accuracy: {accuracy * 100:.2f}%")
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time:.4f} seconds")
            break
        if algorithm_type == "2":
            naive_bayes = NaiveBayes()
            naive_bayes.fit(train_x, train_y)
            nb_predictions = naive_bayes.predict(test_x)
            break
        if algorithm_type == "3":
            perceptron = Perceptron(learning_rate=0.1, epochs=1000)
            perceptron.fit(train_x, train_y)
            perceptron_predictions = perceptron.predict(test_x)
            break
        else:
            print("Invalid input! Please enter 1, 2, or 3.")


# Runs main method
if __name__ == '__main__':
    main()