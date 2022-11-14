import random
class File_selector():
    def __init__(self):
        self.class_0_train = [1,7,8]
        self.class_1_train = [3,4,6,9]
        self.class_0_test = [19,25,26]
        self.class_1_test = [21,22,24,27]

    def select(self):


        picked_class_0_train = self.class_0_train[random.randint(0,len(self.class_0_train)-1)]
        picked_class_1_train = self.class_1_train[random.randint(0,len(self.class_1_train)-1)]


        while True:
            picked_class_0_test = self.class_0_test[random.randint(0,len(self.class_0_test)-1)]
            picked_class_1_test = self.class_1_test[random.randint(0,len(self.class_1_test)-1)]
            if picked_class_0_test != picked_class_0_train+18 and picked_class_1_test != picked_class_1_train+18:
                break
        return str(picked_class_0_train), str(picked_class_1_train), str(picked_class_0_test), str(picked_class_1_test)
        





    
"""
if __name__ == "__file_selecter__":

picked_class_0_train, picked_class_1_train, picked_class_0_test, picked_class_1_test = file_selecter()
print(f"Training Class 0: {picked_class_0_train}\nTraining Class 1: {picked_class_1_train}\nTesting Class 0: {picked_class_0_test}\nTesting Class 1: {picked_class_1_test}\n")
"""