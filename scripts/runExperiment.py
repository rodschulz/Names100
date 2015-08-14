'''
@author: rodrigo
2015
'''
import csv
import shutil
import subprocess

def updateConfigFile(_newValues):
    oldLocation = '../config/config'
    newLocation = '../config/config_new'
    
    oldConfig = open(oldLocation, 'r')
    newConfig = open(newLocation, 'w')
    
    for line in oldConfig:
        if line[0] == '#':
            newConfig.write(line)
        else:
            param = line.replace(' ', '\t').split('\t')
            if param[0] in _newValues:
                newConfig.write(param[0] + '\t\t' + _newValues[param[0]] + '\n')
            else:
                newConfig.write(line)

    oldConfig.close()
    newConfig.close()
    
    shutil.move(newLocation, oldLocation)

###########################
paramsLocation = './params.csv'

with open(paramsLocation, 'rb') as csvfile:
    paramsFile = csv.reader(csvfile, delimiter=';', quotechar='|')
    
    firstRow = True
    
    
    paramNames = []
    for row in paramsFile:
        # store parameters names
        if firstRow:
            firstRow = False
            for param in row:
                paramNames.append(param.replace('\t', ''))
        
        # update config file
        else:
            aux = []
            newValues = {}
            for i in range(len(paramNames)):
                newValues[paramNames[i]] = row[i]
                aux.append(row[i].replace(' ', '').replace('\t', ''))
            
            # update config and execute app
            print('Updating config file')    
            updateConfigFile(newValues)
            
            print('Calling application')
            subprocess.call(['./build/Names ../dataset/'], cwd='../', shell=True)
            
            print('Copying results')
            
            shutil.move('../../results/results', '../../results/' + '-'.join(aux))

print('Finished')
