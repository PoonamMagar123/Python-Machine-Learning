from sklearn import tree

class BallClassifier:
    def __init__(self):
        self.BallFeatures = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]
        self.Names = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]
        self.clf = tree.DecisionTreeClassifier()  # Decide the algorithm
        self.clf = self.clf.fit(self.BallFeatures, self.Names)  # Train the model
    
    def predict_ball(self, weight, surface):
        surface_type = 1 if surface.lower() == "rough" else 0
        result = self.clf.predict([[weight, surface_type]])
        return result[0]

def main():
    print("Ball Classification Case Study")

    print("Enter weight of object: ")
    weight = float(input())

    print("What is the surface type of your object (Rough or Smooth): ")
    surface = input()

    classifier = BallClassifier()
    ball_type = classifier.predict_ball(weight, surface)

    if ball_type == 1:
        print("Your object looks like a Tennis Ball.")
    elif ball_type == 2:
        print("Your object looks like a Cricket Ball.")
    else:
        print("Unable to determine the type of the object.")

if __name__ == "__main__":
    main()
