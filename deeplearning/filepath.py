
import os



def allfiles(path):
    res = []

    for root, dirs, files in os.walk(path):
        rootpath = os.path.join(os.path.abspath(path), root)

        for file in files:
            filepath = os.path.join(rootpath, file)
            res.append(filepath)

    return res

if __name__ == '__main__':

	folder = os.getcwd();
	files = allfiles(folder)

	for file in files:

		temp = file.split('.')
		length = len(temp)
		print temp[length-1]
		print file



