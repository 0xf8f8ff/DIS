import sys



if len(sys.argv) != 2:
    print("Usage: provide number of trees for a Random Forest")
else:
    trees = int(sys.argv[1])    
    with open("dummy", "a") as dummy:
        for i in range(trees):
            dummy.write(str(i+1) + "\n")

