import os

class Test:
    
    # NOTES: For all the testing purpose here
    
    def __init__(self):
        print("Call this class and constructor")
        file_path = os.path.join("test_file", "test.txt")
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file:
            print(file)
        


if __name__ == "__main__":
    test = Test()