import keras.models as Sequential
import keras.layers as Dense
import numpy as np

def main():
    
        data = np.loadtxt("pima-indians-diabetes.csv", delimiter=',')
    
    
        print(data)
    
        model = Sequential()
        model.add(Dense(132, input_dim = 8, activation="relu"))
        model.add(Dense(116, activation="relu"))
        model.add(Dense(108, activation="relu"))
        model.add(Dense(63, activation="relu"))
        model.add(Dense(12, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile()
        model.evaluate()

if __name__ == "__main__":
    main()
