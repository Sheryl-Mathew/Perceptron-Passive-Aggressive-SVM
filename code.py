import datetime as dt
import pandas as pd
import numpy as np
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score,confusion_matrix 
from sklearn.utils import shuffle
import matplotlib.pyplot as lc

f = open('output.txt', 'w')

#To create the log of process execution
def time_stamp(statement):
    print(str(dt.datetime.now())+"\t"+statement, file=f)

#To read dataset
def read_dataset(dataset,type_of_dataset):
    time_stamp("Reading %s Dataset" %(type_of_dataset))
    dataset=pd.read_csv(dataset,header=None)
    if dataset.empty:
        print("%s Dataset is empty" % (type_of_dataset), file=f)
    return dataset

#To split datasets
def split_datasets(train_dataset,dev_dataset,test_dataset):
    time_stamp("Split the training, development and testing dataset into its corresponding feature vectors and class labels")
    x_train=train_dataset.iloc[:,:-1].values
    y_train=train_dataset.iloc[:,9].values
    x_dev=dev_dataset.iloc[:,:-1].values
    y_dev=dev_dataset.iloc[:,9].values
    x_test=test_dataset.iloc[:,:-1].values
    y_test=test_dataset.iloc[:,9].values
    return x_train,y_train,x_dev,y_dev,x_test,y_test

#To combine datasets   
def combine_datasets(datasets,keys_for_datasets,type_of_data):
    time_stamp("Combining %s Datasets" %(type_of_data))
    combined_dataset=pd.concat(datasets,keys=keys_for_datasets)
    return combined_dataset

#To encode feature vectors
def x_encode_dataset(feature_vector):
    time_stamp("Encoding feature vector dataset")
    fv=feature_vector

    for row_num,hpw in enumerate(fv.iloc[:,7]):
        if hpw>=1 and hpw<30:
           fv.iloc[row_num, 7]="Low"
        elif hpw>=30 and hpw<60:
            fv.iloc[row_num, 7]="Medium"
        else:
           fv.iloc[row_num, 7]="High"

    fv=pd.get_dummies(fv,columns=[1,2,3,4,5,6,7,8])

    print(file=f)
    print("We first separate the feature vectors and class labels separately.", file=f)
    print("We convert the continous feature vectors into categorical values. Eg: Hours per week is split into Low, Medium and High.", file=f)
    print("After conversion we use pandas get_dummies function to convert all the categorical feature vectors into binary feature vectors.", file=f)
    print("There are %d features." %(fv.shape[1]), file=f)
    print(file=f)
    return fv

#To encode class labels
def y_encode_dataset(class_labels):
    time_stamp("Encoding class label dataset")
    cl=class_labels
    time_stamp("Income: >50K:1, <=50K:-1")
    cl.iloc[:,0].replace(
        {' >50K':1,
         ' <=50K':-1
        }, inplace=True)
    return cl

#To split combined dataset to train, test, dev
def split_combined_dataset(dataset,type_of_dataset):
    time_stamp("Split the combined %s dataset into its corresponding training, development and testing datasets" %(type_of_dataset))
    train, dev, test = dataset.xs('train'), dataset.xs('dev'), dataset.xs('test')
    return train, dev, test

#To initialise values to zero
def initialise_vectors_zero(zero_vector):
    result=np.zeros((zero_vector.shape[1]))
    return result

#To calculate learning rate
def calculate_learning_rate(feature_vector,class_label,weights):
    x=feature_vector
    y=class_label
    w=weights
    numerator = 1 - y * np.dot(w, x)
    denominator = np.square(np.linalg.norm(x))
    learning_rate = numerator / denominator
    return learning_rate

#Standard Perceptron
def perceptron_algorithm(x_train,y_train,x_dev,y_dev,x_test,y_test,max_iter,learning_rate,plot_curves):
    time_stamp("Standard Perceptron Algorithm")

    w = initialise_vectors_zero(x_train)
    x = x_train
    y = y_train
    total_mistakes = []
    total_iterations = []
    total_examples = x.shape[0]
    train_accuracies = []
    dev_accuracies = []
    test_accuracies = []
    
    for i in range(max_iter):
        mistakes=0
        for j in range(total_examples):
            y_hat = np.sign(np.dot(x[j],w))
            if y_hat == 0:
                y_hat = -1
            if y_hat != y[j]:
                mistakes = mistakes + 1
                w = w + learning_rate * np.dot(y[j],x[j])
     
        total_mistakes.append(mistakes)
        total_iterations.append(i + 1)

        train_accuracy = calculate_accuracy(mistakes,total_examples)
        dev_accuracy = testing_algorithm(x_dev,y_dev,w,0)
        test_accuracy = testing_algorithm(x_test,y_test,w,0)
        
        train_accuracies.append(train_accuracy)
        dev_accuracies.append(dev_accuracy)
        test_accuracies.append(test_accuracy)

    if plot_curves == "Yes":
        plot_learning_curve(total_mistakes,total_iterations,"Perceptron","Perceptron_Learning_Curve.png")
        plot_accuracy_curve(total_iterations,train_accuracies,"Number of Iterations","Perceptron Training Data","Perceptron_Training_Accuracy_Curve.png")
        plot_accuracy_curve(total_iterations,dev_accuracies,"Number of Iterations","Perceptron Development Data","Perceptron_Development_Accuracy_Curve.png")
        plot_accuracy_curve(total_iterations,test_accuracies,"Number of Iterations","Perceptron Testing Data","Perceptron_Testing_Accuracy_Curve.png")

    final_train_accuracy = calculate_accuracy(mistakes,total_examples)
    final_dev_accuracy = testing_algorithm(x_dev,y_dev,w,0)
    final_test_accuracy = testing_algorithm(x_test,y_test,w,0)

    print(file=f)
    print("Training Accuracy = %0.2f" %(final_train_accuracy), file=f)
    print("Development Accuracy = %0.2f" %(final_dev_accuracy), file=f)
    print("Test Accuracy = %0.2f" %(final_test_accuracy), file=f)
    print(file=f)

    return final_train_accuracy, final_dev_accuracy, final_test_accuracy

#Passive Aggressive
def passive_aggressive_algorithm(x_train,y_train,x_dev,y_dev,x_test,y_test,max_iter,plot_curves):
    time_stamp("Passive Aggressive Algorithm")
    w = initialise_vectors_zero(x_train)
    x = x_train
    y = y_train
    total_mistakes = []
    total_iterations = []
    total_examples = x.shape[0]
    train_accuracies = []
    dev_accuracies = []
    test_accuracies = []

    for i in range(max_iter):
        mistakes=0
        for j in range(total_examples):
            y_hat = np.sign(np.dot(w,x[j]))
            if y_hat == 0:
                y_hat = -1
            if y[j] != y_hat:
                learning_rate = calculate_learning_rate(x[j],y[j],w)
                mistakes = mistakes + 1
                w = w + learning_rate * np.dot(y[j],x[j])

        total_mistakes.append(mistakes)
        total_iterations.append(i + 1)

        train_accuracy = calculate_accuracy(mistakes,total_examples)
        dev_accuracy = testing_algorithm(x_dev,y_dev,w,0)
        test_accuracy = testing_algorithm(x_test,y_test,w,0)
        
        train_accuracies.append(train_accuracy)
        dev_accuracies.append(dev_accuracy)
        test_accuracies.append(test_accuracy)

    if plot_curves == "Yes":
        plot_learning_curve(total_mistakes,total_iterations,"Passive Aggressive","Passive_Aggressive_Learning_Curve.png")
        plot_accuracy_curve(total_iterations,train_accuracies,"Number of Iterations","Passive Aggressive Training Data","Passive_Aggressive_Training_Accuracy_Curve.png")
        plot_accuracy_curve(total_iterations,dev_accuracies,"Number of Iterations","Passive Aggressive Development Data","Passive_Aggressive_Development_Accuracy_Curve.png")
        plot_accuracy_curve(total_iterations,test_accuracies,"Number of Iterations","Passive Aggressive Testing Data","Passive_Aggressive_Testing_Accuracy_Curve.png")

    final_train_accuracy = calculate_accuracy(mistakes,total_examples)
    final_dev_accuracy = testing_algorithm(x_dev,y_dev,w,0)
    final_test_accuracy = testing_algorithm(x_test,y_test,w,0)

    print(file=f)
    print("Training Accuracy = %0.2f" %(final_train_accuracy), file=f)
    print("Development Accuracy = %0.2f" %(final_dev_accuracy), file=f)
    print("Test Accuracy = %0.2f" %(final_test_accuracy), file=f)
    print(file=f)

    return final_train_accuracy, final_dev_accuracy, final_test_accuracy

#Average Perceptron Naive
def avg_perceptron_naive_algorithm(x_train,y_train,x_dev,y_dev,x_test,y_test,max_iter,learning_rate):
    time_stamp("Average Perceptron Naive Algorithm")
    w = initialise_vectors_zero(x_train)
    x = x_train
    y = y_train
    total_examples = x.shape[0]
    w_sum = 0
    start_time = dt.datetime.now()

    for i in range(max_iter):
        mistakes=0
        for j in range(total_examples):
            y_hat = np.sign(np.dot(x[j],w))
            if y_hat == 0:
                y_hat = -1
            if y_hat != y[j]:
                mistakes = mistakes + 1
                w = w + learning_rate * np.dot(y[j],x[j])
                w_sum = w + w_sum
    w_avg = w_sum/total_examples

    end_time = dt.datetime.now()
    time_taken = (end_time - start_time).total_seconds()
    print(file=f)
    print("Execution time for Average Perceptron Naive Algorithm = %f"%(time_taken), file=f)
      
    print(file=f)
    print("Training Accuracy = %0.2f" %(calculate_accuracy(mistakes,total_examples)), file=f)
    print("Development Accuracy = %0.2f" %(testing_algorithm(x_dev,y_dev,w_avg,0)), file=f)
    print("Test Accuracy = %0.2f" %(testing_algorithm(x_test,y_test,w_avg,0)), file=f)
    print(file=f)

#Average Perceptron Hal
def avg_perceptron_hal_algorithm(x_train,y_train,x_dev,y_dev,x_test,y_test,max_iter,learning_rate):
    time_stamp("Average Perceptron Hal Algorithm")
    w = initialise_vectors_zero(x_train)
    u = initialise_vectors_zero(x_train)
    c = 1
    b = 0
    beta = 0
    x = x_train
    y = y_train
    total_examples = x.shape[0]
    start_time = dt.datetime.now()

    for i in range(max_iter):
        mistakes=0
        for j in range(total_examples):
            y_hat = y[j] * (np.dot(x[j],w) + b)
            if y_hat <=0:
                mistakes = mistakes + 1
                w = w + learning_rate * np.dot(y[j],x[j])
                b = b + y[j]
                u = u + c * np.dot(y[j],x[j])
                beta = beta + y[j] * c
            c = c + 1
    w_avg = w - (u/c)
    bias = b - (beta/c)
        
    end_time = dt.datetime.now()
    time_taken = (end_time - start_time).total_seconds()
    print(file=f)
    print("Execution time for Average Perceptron Hal Algorithm = %f"%(time_taken), file=f)
    print(file=f)
    
    print("Training Accuracy = %0.2f" %(calculate_accuracy(mistakes,total_examples)), file=f)
    print("Development Accuracy = %0.2f" %(testing_algorithm(x_dev,y_dev,w_avg,bias)), file=f)
    print("Test Accuracy = %0.2f" %(testing_algorithm(x_test,y_test,w_avg,bias)), file=f)
    print(file=f)

#Standard Perceptron General Learning
def perceptron_algorithm_glc(x_train,y_train,x_dev,y_dev,x_test,y_test):
    time_stamp("Standard Perceptron General Learning Algorithm")
    w = initialise_vectors_zero(x_train)
    x = x_train
    y = y_train
    total_mistakes = []
    total_iterations = []
    total_examples = x.shape[0]
    train_accuracies = []
    dev_accuracies = []
    test_accuracies = []
    row_range = [5000,10000,15000,20000,25000]

    for rows in row_range:
        print(file=f)
        time_stamp("Number of examples = %d" %(rows))
        x_subset_train = x[:rows,:]
        y_subset_train = y[:rows]
        train_accuracy,dev_accuracy,test_accuracy = perceptron_algorithm(x_subset_train, y_subset_train, x_dev,y_dev, x_test,y_test, 5,1,"No")
        train_accuracies.append(train_accuracy)
        dev_accuracies.append(dev_accuracy)
        test_accuracies.append(test_accuracy)
    plot_general_learning_curve(row_range,train_accuracies,dev_accuracies,test_accuracies,"Perceptron","Perceptron_General_Learning_Curve.png")

#Passive Aggressive General Learning
def passive_aggressive_algorithm_glc(x_train,y_train,x_dev,y_dev,x_test,y_test):
    time_stamp("Passsive Aggressive General Learning Algorithm")
    w = initialise_vectors_zero(x_train)
    x = x_train
    y = y_train
    total_mistakes = []
    total_iterations = []
    total_examples = x.shape[0]
    train_accuracies = []
    dev_accuracies = []
    test_accuracies = []
    row_range = [5000,10000,15000,20000,25000]
    
    for rows in row_range:
        print(file=f)
        time_stamp("Number of examples = %d" %(rows))
        x_subset_train = x[:rows,:]
        y_subset_train = y[:rows]                                                                                
        train_accuracy,dev_accuracy,test_accuracy = passive_aggressive_algorithm(x_subset_train,y_subset_train,x_dev,y_dev,x_test,y_test,5,"No")
        train_accuracies.append(train_accuracy)
        dev_accuracies.append(dev_accuracy)
        test_accuracies.append(test_accuracy)
    plot_general_learning_curve(row_range,train_accuracies,dev_accuracies,test_accuracies,"Passive Aggressive","Passive_Aggressive_General_Learning_Curve.png")

#Standard Perceptron Variable Learning Rate
def perceptron_algorithm_vary_learning_rate(x_train,y_train,x_dev,y_dev,x_test,y_test,max_iter):
    time_stamp("Standard Perceptron Algorithm Varying Learning Rate")

    w = initialise_vectors_zero(x_train)
    x = x_train
    y = y_train
    total_mistakes = []
    total_iterations = []
    total_examples = x.shape[0]
    train_accuracies = []
    dev_accuracies = []
    test_accuracies = []
    learning_rate = 1

    for i in range(max_iter):
        mistakes=0
        time_stamp("Learning Rate = %.2f" %(learning_rate))
        for j in range(total_examples):
            y_hat = np.sign(np.dot(x[j],w))
            if y_hat == 0:
                y_hat = -1
            if y_hat != y[j]:
                mistakes = mistakes + 1
                w = w + learning_rate * np.dot(y[j],x[j])
        learning_rate = learning_rate - 0.1
        total_mistakes.append(mistakes)
        total_iterations.append(i + 1)

        train_accuracy = calculate_accuracy(mistakes,total_examples)
        dev_accuracy = testing_algorithm(x_dev,y_dev,w,0)
        test_accuracy = testing_algorithm(x_test,y_test,w,0)
        
        train_accuracies.append(train_accuracy)
        dev_accuracies.append(dev_accuracy)
        test_accuracies.append(test_accuracy)

   
    plot_learning_curve(total_mistakes,total_iterations,"Perceptron Varying Learning Rate","Perceptron_VLR_Learning_Curve.png")
    plot_accuracy_curve(total_iterations,train_accuracies,"Number of Iterations","Perceptron  Varying Learning Rate Training Data","Perceptron_VLR_Training_Accuracy_Curve.png")
    plot_accuracy_curve(total_iterations,dev_accuracies,"Number of Iterations","Perceptron  Varying Learning Rate Development Data","Perceptron_VLR_Development_Accuracy_Curve.png")
    plot_accuracy_curve(total_iterations,test_accuracies,"Number of Iterations","Perceptron  Varying Learning Rate Testing Data","Perceptron_VLR_Testing_Accuracy_Curve.png")

    final_train_accuracy = calculate_accuracy(mistakes,total_examples)
    final_dev_accuracy = testing_algorithm(x_dev,y_dev,w,0)
    final_test_accuracy = testing_algorithm(x_test,y_test,w,0)

    print(file=f)
    print("Training Accuracy = %0.2f" %(final_train_accuracy), file=f)
    print("Development Accuracy = %0.2f" %(final_dev_accuracy), file=f)
    print("Test Accuracy = %0.2f" %(final_test_accuracy), file=f)
    print(file=f)

    return final_train_accuracy, final_dev_accuracy, final_test_accuracy

#Standard Perceptron Shuffle Examples
def perceptron_algorithm_shuffle(x_train,y_train,x_dev,y_dev,x_test,y_test,max_iter,learning_rate):
    time_stamp("Standard Perceptron Algorithm Shuffle Examples")

    w = initialise_vectors_zero(x_train)
    x = x_train
    y = y_train
    total_mistakes = []
    total_iterations = []
    total_examples = x.shape[0]
    train_accuracies = []
    dev_accuracies = []
    test_accuracies = []
    
    for i in range(max_iter):
        mistakes=0
        time_stamp("Shuffle Dataframe")
        x, y = shuffle(x, y)  
        for j in range(total_examples):
            y_hat = np.sign(np.dot(x[j],w))
            if y_hat == 0:
                y_hat = -1
            if y_hat != y[j]:
                mistakes = mistakes + 1
                w = w + learning_rate * np.dot(y[j],x[j])
     
        total_mistakes.append(mistakes)
        total_iterations.append(i + 1)

        train_accuracy = calculate_accuracy(mistakes,total_examples)
        dev_accuracy = testing_algorithm(x_dev,y_dev,w,0)
        test_accuracy = testing_algorithm(x_test,y_test,w,0)
        
        train_accuracies.append(train_accuracy)
        dev_accuracies.append(dev_accuracy)
        test_accuracies.append(test_accuracy)

    plot_learning_curve(total_mistakes,total_iterations,"Perceptron Shuffling Examples","Perceptron_Shuffle_Learning_Curve.png")
    plot_accuracy_curve(total_iterations,train_accuracies,"Number of Iterations","Perceptron Shuffling Examples Training Data","Perceptron_Shuffle_Training_Accuracy_Curve.png")
    plot_accuracy_curve(total_iterations,dev_accuracies,"Number of Iterations","Perceptron Shuffling Examples Development Data","Perceptron_Shuffle_Development_Accuracy_Curve.png")
    plot_accuracy_curve(total_iterations,test_accuracies,"Number of Iterations","Perceptron Shuffling Examples Testing Data","Perceptron_Shuffle_Testing_Accuracy_Curve.png")

    final_train_accuracy = calculate_accuracy(mistakes,total_examples)
    final_dev_accuracy = testing_algorithm(x_dev,y_dev,w,0)
    final_test_accuracy = testing_algorithm(x_test,y_test,w,0)

    print(file=f)
    print("Training Accuracy = %0.2f" %(final_train_accuracy), file=f)
    print("Development Accuracy = %0.2f" %(final_dev_accuracy), file=f)
    print("Test Accuracy = %0.2f" %(final_test_accuracy), file=f)
    print(file=f)

    return final_train_accuracy, final_dev_accuracy, final_test_accuracy

#Testing Algorithm
def testing_algorithm(feature_vector,class_label,weights,bias):
    x = feature_vector
    y = class_label
    w = weights
    mistakes = 0
    total_examples = x.shape[0]
    b = bias
    for j in range(total_examples):
        y_hat = np.sign(np.dot(x[j],w)+b)
        if y_hat == 0:
            y_hat = -1
        if y_hat != y[j]:
            mistakes = mistakes + 1
    accuracy = calculate_accuracy(mistakes,total_examples)
    return accuracy

#To calculate accuracy
def calculate_accuracy(mistakes,total_examples):
    error_rate = mistakes/total_examples
    accuracy = (1 - error_rate)*100
    return accuracy

#To plot learning curves
def plot_learning_curve(mistakes,iterations,type_of_plot,save_file_name):
    time_stamp("Plotting learning curve for %s" %(type_of_plot))
    lc.plot(iterations, mistakes, color = 'blue', marker='o', linestyle='solid')
    lc.xlabel('Number of Iterations')
    lc.ylabel('Number of Mistakes')
    lc.title("Learning curve for %s" %(type_of_plot))
    lc.savefig(save_file_name)
    lc.show()

#To plot accuracy curves
def plot_accuracy_curve(x_values,accuracies,x_label,type_of_plot,save_file_name):
    time_stamp("Plotting accuracy curve for %s" %(type_of_plot))
    lc.plot(x_values, accuracies, color = 'blue', marker='o', linestyle='solid')
    lc.xlabel(x_label)
    lc.ylabel('Accuracy (%)')
    lc.title("Accuracy curve for %s" %(type_of_plot))
    lc.savefig(save_file_name)
    lc.show()

#To plot general learning curves
def plot_general_learning_curve(x_values,y_value_train,y_value_dev,y_value_test,type_of_plot,save_file_name):
    time_stamp("Plotting general learning curve for %s" %(type_of_plot))
    lc.plot(x_values, y_value_train, color = 'red', marker='o', linestyle='solid', label='Training Data')
    lc.plot(x_values, y_value_dev, color = 'green', marker='o', linestyle='solid', label='Development Data')
    lc.plot(x_values, y_value_test, color = 'blue', marker='o', linestyle='solid', label='Testing Data')
    lc.legend(loc='upper left')
    lc.xlabel("Number of Examples")
    lc.ylabel("Accuracy (%)")
    lc.title("General learning curve for %s" %(type_of_plot))
    lc.savefig(save_file_name)
    lc.show()

#SVM Classifier for Linear Kernel based on C values
def svm_classifier_c_parameter(x_train,y_train,x_dev,y_dev,x_test,y_test):
    time_stamp("SVM Classifier for Linear Kernel based on C values")
    train_accuracies=[]
    test_accuracies=[]
    dev_accuracies=[]
    support_vectors=[]
    c_values = [0.0001,0.001,0.01,0.1,1,10]
    for c in c_values:
        print(file=f)
        time_stamp("SVM for C = %f" %(c))
        svm_classifier = SVC(kernel='linear',C=c)
        svm_classifier.fit(x_train, y_train)
        y_train_predictions = svm_classifier.predict(x_train)
        y_dev_predictions = svm_classifier.predict(x_dev)
        y_test_predictions = svm_classifier.predict(x_test)
        number_of_support_vectors = sum(svm_classifier.n_support_)
        train_accuracy = accuracy_score(y_train,y_train_predictions) *100
        dev_accuracy = accuracy_score(y_dev,y_dev_predictions)*100
        test_accuracy = accuracy_score(y_test,y_test_predictions)*100
        print(file=f)
        print("Training Accuracy = %.2f" %(train_accuracy), file=f)
        print("Validation Accuracy =  %.2f" %(dev_accuracy), file=f)
        print("Testing Accuracy = %.2f" %(test_accuracy), file=f)
        print("Number of Support Vectors = %d" %(number_of_support_vectors), file=f)
        train_accuracies.append(train_accuracy)
        dev_accuracies.append(dev_accuracy)
        test_accuracies.append(test_accuracy)
        support_vectors.append(number_of_support_vectors)
    max_dev_accuracy=max(dev_accuracies)
    index_max_dev_accuracy=dev_accuracies.index(max_dev_accuracy)
    best_c=c_values[index_max_dev_accuracy]
    print(file=f)
    print("The best value of hyper-parameter C based on the accuracy on validation set = %f" %(best_c), file=f)
    print(file=f)
    return c_values,train_accuracies,dev_accuracies,test_accuracies,support_vectors,best_c

#SVM Classifier for Combined Training and Validation data and Confusion Matrix
def svm_classifier_confusion_matrix(x_train,y_train,x_dev,y_dev,x_test,y_test,best_c):
    time_stamp("SVM for Combined Training and Validation data and Confusion Matrix")
    x_combined = pd.concat([pd.DataFrame(x_train),pd.DataFrame(x_dev)])
    y_concat = pd.concat([pd.DataFrame(y_train),pd.DataFrame(y_dev)])
    y_combined = np.ravel(y_concat)
    svm_classifier = SVC(kernel='linear', C=best_c)
    svm_classifier.fit(x_combined, y_combined)
    y_train_predictions = svm_classifier.predict(x_combined)
    y_test_predictions = svm_classifier.predict(x_test)
    number_of_support_vectors = sum(svm_classifier.n_support_)
    train_accuracy = accuracy_score(y_combined,y_train_predictions) *100
    test_accuracy = accuracy_score(y_test,y_test_predictions)*100
    train_confusion_matrix = confusion_matrix(y_combined,y_train_predictions)
    test_confusion_matrix = confusion_matrix(y_test,y_test_predictions)
    print(file=f)
    print("Training Accuracy : %.2f" %(train_accuracy), file=f)
    print("Testing Accuracy : %.2f" %(test_accuracy), file=f)
    print("Number of Support Vectors : %d" %(number_of_support_vectors), file=f)
    print("Confusion matrix for Training",file=f)
    print(train_confusion_matrix, file=f)
    print("Confusion matrix for Testing", file=f)
    print(test_confusion_matrix, file=f)
    print(file=f)

#SVM Classifier for different kernels
def svm_classifier_kernels(x_train,y_train,x_dev,y_dev,x_test,y_test,best_c):
    time_stamp("SVM Classifier for different kernels")
    train_accuracies=[]
    test_accuracies=[]
    dev_accuracies=[]
    support_vectors=[]
    degree_of_polynomial = [2,3,4]
    print(file=f)
    time_stamp("SVM for Linear Kernel")
    svm_classifier = SVC(kernel='linear',C=best_c)
    svm_classifier.fit(x_train, y_train)
    y_train_predictions = svm_classifier.predict(x_train)
    y_dev_predictions = svm_classifier.predict(x_dev)
    y_test_predictions = svm_classifier.predict(x_test)
    number_of_support_vectors = sum(svm_classifier.n_support_)
    train_accuracy = accuracy_score(y_train,y_train_predictions) *100
    dev_accuracy = accuracy_score(y_dev,y_dev_predictions)*100
    test_accuracy = accuracy_score(y_test,y_test_predictions)*100
    print(file=f)
    print("Training Accuracy : %.2f" %(train_accuracy), file=f)
    print("Validation Accuracy : %.2f" %(dev_accuracy), file=f)
    print("Testing Accuracy : %.2f" %(test_accuracy), file=f)
    print("Number of Support Vectors : %d" %(number_of_support_vectors), file=f)
    print(file=f)
    train_accuracies.append(train_accuracy)
    dev_accuracies.append(dev_accuracy)
    test_accuracies.append(test_accuracy)
    support_vectors.append(number_of_support_vectors)

    for degree in degree_of_polynomial:
        time_stamp("SVM for Polynomial Kernel of degree %d" %(degree))
        svm_classifier = SVC(kernel='poly', degree=degree, C=best_c)
        svm_classifier.fit(x_train, y_train)
        y_train_predictions = svm_classifier.predict(x_train)
        y_dev_predictions = svm_classifier.predict(x_dev)
        y_test_predictions = svm_classifier.predict(x_test)
        number_of_support_vectors = sum(svm_classifier.n_support_)
        train_accuracy = accuracy_score(y_train,y_train_predictions) *100
        dev_accuracy = accuracy_score(y_dev,y_dev_predictions)*100
        test_accuracy = accuracy_score(y_test,y_test_predictions)*100
        print(file=f)
        print("Training Accuracy = %.2f" %(train_accuracy), file=f)
        print("Validation Accuracy =  %.2f" %(dev_accuracy), file=f)
        print("Testing Accuracy = %.2f" %(test_accuracy), file=f)
        print("Number of Support Vectors = %d" %(number_of_support_vectors), file=f)
        print(file=f)
        train_accuracies.append(train_accuracy)
        dev_accuracies.append(dev_accuracy)
        test_accuracies.append(test_accuracy)
        support_vectors.append(number_of_support_vectors)
    max_dev_accuracy=max(dev_accuracies)
    index_max_dev_accuracy=dev_accuracies.index(max_dev_accuracy)
    best_degree=degree_of_polynomial[index_max_dev_accuracy]
    print(file=f)
    print("The best value of degree for polynomial kernel = %f" %(best_degree), file=f)
    print(file=f)
    return train_accuracies,dev_accuracies,test_accuracies,support_vectors,best_degree

#To plot support vectors
def plot_support_vector(c_values,support_vectors,type_of_data,save_file_name):
    time_stamp("Plotting of number of support vectors for different values of %s" %(type_of_data))
    lc.plot(c_values, support_vectors, color = 'blue', marker='o', linestyle='solid')
    lc.xlabel(type_of_data)
    lc.ylabel("Number of support vectors")
    lc.title("Number of support vectors for different values of %s" %(type_of_data))
    lc.savefig(save_file_name)
    lc.show()

#Get Dataset
train_dataset = 'income.train.txt'
dev_dataset   = 'income.dev.txt'
test_dataset  = 'income.test.txt'

#Read datasets
read_train_dataset = read_dataset(train_dataset,"training")
read_dev_dataset   = read_dataset(dev_dataset,"development")
read_test_dataset  = read_dataset(test_dataset,"testing")

#Split Dataset
x_train_split,y_train_split,x_dev_split,y_dev_split,x_test_split,y_test_split = split_datasets(read_train_dataset,read_dev_dataset,read_test_dataset)
x_train_pd = pd.DataFrame(x_train_split)
x_dev_pd   = pd.DataFrame(x_dev_split)
x_test_pd  = pd.DataFrame(x_test_split)
y_train_pd = pd.DataFrame(y_train_split)
y_dev_pd   = pd.DataFrame(y_dev_split)
y_test_pd  = pd.DataFrame(y_test_split)

#Combine dataset
x_combined_dataset = combine_datasets([x_train_pd,x_dev_pd,x_test_pd],['train','dev','test'],"feature vector")
y_combined_dataset = combine_datasets([y_train_pd,y_dev_pd,y_test_pd],['train','dev','test'],"class label")

#Encode dataset
x_encoded_dataset = x_encode_dataset(x_combined_dataset)
y_encoded_dataset = y_encode_dataset(y_combined_dataset)

#Split Combined Dataset
x_train_encoded,x_dev_encoded,x_test_encoded = split_combined_dataset(x_encoded_dataset,"feature vector")
y_train_encoded,y_dev_encoded,y_test_encoded = split_combined_dataset(y_encoded_dataset,"class label")

#Getting Feature vectors and Class labels
x_train = x_train_encoded.iloc[:, :].values
x_dev   = x_dev_encoded.iloc[:, :].values
x_test  = x_test_encoded.iloc[:, :].values
y_train = np.ravel(y_train_encoded)
y_dev   = np.ravel(y_dev_encoded)
y_test  = np.ravel(y_test_encoded)

# Online Learning Algorithms 

perceptron_algorithm(x_train,y_train,x_dev,y_dev,x_test,y_test,5,1,"Yes")
passive_aggressive_algorithm(x_train,y_train,x_dev,y_dev,x_test,y_test,5,"Yes")
print(file=f)
print("Comparison of Online Learning curves for Perceptron and Passive Aggressive",file=f)
print("Perceptron at its 1st iteration has 7600 mistakes, 2nd iteration 6700 mistakes and so on. Passive Aggressive at its 1st iteration has 7700 mistakes, 2nd iteration 7000 mistakes and so on. Therefore Perceptron Algorithm learns faster when compared to Passive aggressive for few iterations.",file=f)
print(file=f)
print("Comparison of Accuracy Curves for Perceptron and Passive Aggressive",file=f)
print("The accuracy curve for training data of perceptron is higher compared to that of passive aggressive for every iteration. But for development and test data passive aggressive has an increase in the accuracy for every iteration. But for perceptron there is an increase in the accuracies and a sharp drop in accuracy when number of iterations increases.",file=f)
print(file=f)
avg_perceptron_naive_algorithm(x_train,y_train,x_dev,y_dev,x_test,y_test,5,1)
avg_perceptron_hal_algorithm(x_train,y_train,x_dev,y_dev,x_test,y_test,5,1)
print(file=f)
print("Compare Accuracies for Perceptron and Average Perceptron",file=f)
print("We notice that the accuracies for training, development and test data is higher for either of the Average Perceptron implementation when compared to Standard Perceptron. But the accuracies for training, development and test data Average Perceptron Hal implementation is higher compared to the Naive Algorithm.",file=f)
print(file=f)
perceptron_algorithm_glc(x_train,y_train,x_dev,y_dev,x_test,y_test)
passive_aggressive_algorithm_glc(x_train,y_train,x_dev,y_dev,x_test,y_test)
print(file=f)
print("Comparison of General Learning Curve for Perceptron and Passive Aggressive",file=f)
print("The general learning curve for train of both Perceptron and Passive Aggressive does not change by much when number of examples are increased for every iteration. In perceptron for development and testing data when the number of examples increases initially there is a rise and then drop in accuracy but for passive aggressive initially there is a drop in accuracy when number of iterations increase but afterwards there is a steady increase.",file=f)
print(file=f)
perceptron_algorithm_vary_learning_rate(x_train,y_train,x_dev,y_dev,x_test,y_test,5)
print(file=f)
print("Observing Perceptron when training examples are shuffled",file=f)
print("The learning rate is reduced by 0.1 for every iteration. The online learning curve has almost the same number of mistakes for iteration 4 and 5 (almost converge). But the accuracy curve has increased for every iteration without any drop in accuracy when iterations increases. The accuracy has also increased when compared to standard perceptron.",file=f)
print(file=f)
perceptron_algorithm_shuffle(x_train,y_train,x_dev,y_dev,x_test,y_test,5,1)
print(file=f)
print("Observing Perceptron when training examples are shuffled",file=f)
print("The training examples are shuffled every iteration. The online learning curve is varying for every iteration. But the accuracy curve is similar to standard perceptron. Accuracy for training is less while development and test accuracy has increased when compared to standard perceptron.",file=f)
print(file=f)

# SVM Classification

c_values,train_accuracies_c,dev_accuracies_c,test_accuracies_c,support_vectors_c,best_c=svm_classifier_c_parameter(x_train,y_train,x_dev,y_dev,x_test,y_test)
plot_accuracy_curve(c_values,train_accuracies_c,"training of linear kernel for various values of C","C for training data","Accuracy_Curve_C_Training.png")
plot_accuracy_curve(c_values,dev_accuracies_c,"development of linear kernel for various values of C","C for development data","Accuracy_Curve_C_Development.png")
plot_accuracy_curve(c_values,test_accuracies_c,"testing of linear kernel for various values of C for testing data","C","Accuracy_Curve_C_Testing.png")
plot_support_vector(c_values,support_vectors_c,"C","Support_Vector_C.png")
print(file=f)
print("Observations for Varying C",file=f)
print("In the accuracy curve, the accuracy for 0.0001 to 0.01 is increasing after that it is almost constant. In the support vectors curve, the number of support vectors for 0.0001 to 0.01 is decreasing after that it is almost constant.", file=f)
print(file=f)

svm_classifier_confusion_matrix(x_train,y_train,x_dev,y_dev,x_test,y_test,best_c)

train_accuracies_k,dev_accuracies_k,test_accuracies_k,support_vectors_k,best_degree=svm_classifier_kernels(x_train,y_train,x_dev,y_dev,x_test,y_test,best_c)
plot_accuracy_curve(["linear","Polynomial 2","Polynomial 3","Polynomial 4"],train_accuracies_k,"Kernel Types","training of different kernels","Accuracy_Curve_Kernel_Training.png")
plot_accuracy_curve(["linear","Polynomial 2","Polynomial 3","Polynomial 4"],dev_accuracies_k,"Kernel Types","development of different kernels","Accuracy_Curve_Kernel_Development.png")
plot_accuracy_curve(["linear","Polynomial 2","Polynomial 3","Polynomial 4"],test_accuracies_k,"Kernel Types","testing of different kernels","Accuracy_Curve_Kernel_Testing.png")
plot_support_vector(["Linear","Polynomial 2","Polynomial 3","Polynomial 4"],support_vectors_k,"Kernel Types","Support_Vector_Kernel_Types.png")
print(file=f)
print("Observations for Different Kernels",file=f)
print("Polynomial of Degree 3 Kernel has the highest development and testing accuracy. While Polynomial of degree 4 Kernel has the least supprt vectors.", file=f)
print(file=f)

f.close()