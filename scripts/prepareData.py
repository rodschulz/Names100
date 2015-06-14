'''
@author: rodrigo
2015
'''
import sys
import os
import shutil
import random


def createSet(dataLocation, outputFolder, classes, className, setName, setSize):
    print('Creating set "' + setName + '" for class "' + className + '"')
    
    # get set's data
    dataset = random.sample(classes[className], setSize)
    
    # create new folders
    classFolder = outputFolder + className + '/'
    if not os.path.exists(classFolder): 
        os.makedirs(classFolder)
    
    destinationFolder = classFolder + className + '_' + setName + '/'
    if not os.path.exists(destinationFolder): 
        os.makedirs(destinationFolder)
    
    # copy data
    for item in dataset:
        classes[className].remove(item)
        shutil.copyfile(dataLocation + item, destinationFolder + item)
   
     
'''
<dataset_location>    folder where the images are
<output_folder>       folder where the resulting sets will be stored
<train_size>          size of the train set (as decimal number, ie 0.3 == 30%)
<validation_size>     size of the validation set (as decimal number, ie 0.3 == 30%)
'''    

if (len(sys.argv) < 4):
    print('Not enough arguments.')
    print('Usage:\n\t<dataset_location> <output_folder> <train_size> <validation_size>')
    sys.exit()

classSize = 800

dataLocation = sys.argv[1]
outputFolder = sys.argv[2]
trainSize = int(float(sys.argv[3]) * classSize)
validationSize = int(float(sys.argv[4]) * classSize)
testSize = classSize - (trainSize + validationSize)

if testSize <= 0:
    print('ERROR: invalid sizes for train and/or validation set. Quiting.')
    sys.exit()

# remove previous data
if os.path.exists(outputFolder):
    print('Removing previous data')
    shutil.rmtree(outputFolder)

# create output folder
print('Creating output folder in ' + outputFolder)
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

# extract clases and elements
print('Parsing files')
classes = {}
for f in os.listdir(dataLocation):
    if (os.path.isfile(dataLocation + f)):
        data = f.split('_')
        
        if not data[0] in classes:
            classes[data[0]] = []
        classes[data[0]].append(f)

# create sets
for name in classes:
    createSet(dataLocation, outputFolder, classes, name, 'train', trainSize)
    createSet(dataLocation, outputFolder, classes, name, 'val', validationSize)
    createSet(dataLocation, outputFolder, classes, name, 'test', testSize)

print('Finished')