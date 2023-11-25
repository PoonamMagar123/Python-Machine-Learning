
from sklearn import tree

def MarvellousML(weight,surface):
    BallFeatures = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1], [92,0],[35,1],[35,1],[35,1], [96,0], [43,1],[110,0], [35,1],[95,0]]

    Names = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

    cif = tree.DecisionTreeClassifier()     # Decide the algorithm

    cif = cif.fit(BallFeatures,Names)      # Train the model
    
    result = cif.predict([[weight,surface]])
    
    if result == 1:
        print("Your Object Looks like Tennis Ball")
    elif result == 2:
        print("Your Object Looks like Cricket Ball")

def main():
    print("Ball Classification Case Study")

    print("Enter weight of object : ")
    weight = input()
    
    print("What is the surface type of Your object Rough or Smooth")
    surface = input()
    
    if surface.lower() == "rough":
        surface = 1
    elif surface.lower() == "smooth":
        surface = 0
    else:
        print("Error : Wrong input")
        exit()
    
    MarvellousML(weight, surface)
    
if __name__ == "__main__":
    main()