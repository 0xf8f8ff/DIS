
with open("dummy", "a") as dummy:
    for i in range(100):
        dummy.write(str(i+1) + "\n")