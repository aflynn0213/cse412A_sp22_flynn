# buyLotsOfFruit.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
To run this script, type

  python buyLotsOfFruit.py
  
Once you have correctly implemented the buyLotsOfFruit function,
the script should produce the output:

Cost of [('apples', 2.0), ('pears', 3.0), ('limes', 4.0)] is 12.25
"""

fruitPrices = {'apples':2.00, 'oranges': 1.50, 'pears': 1.75,
              'limes':0.75, 'strawberries':1.00}

def buyLotsOfFruit(orderList):
    """
        orderList: List of (fruit, numPounds) tuples
            
    Returns cost of order
    """ 
    totalCost = 0.0             
    validFruit = fruitPrices.keys() 

    for k in orderList:
	currFruit = k[0]
        check = checkList(validFruit,currFruit)
     	if (check == False):
	    print "ERROR INVALID FRUIT ENTRY"
	    return "None"
	
	totalCost += fruitPrices[currFruit]*k[1]	
    
    return totalCost

def checkList(fruitList,fruit):
    
    for i in range(len(fruitList)):
	if (fruitList[i] == fruit):
	    return True
    return False
	   
# Main Method    
if __name__ == '__main__':
    "This code runs when you invoke the script from the command line"
    orderList = [ ('apples', 2.0), ('pears', 3.0), ('limes', 4.0) ]
    cost = buyLotsOfFruit(orderList)
    print 'Cost of', orderList, 'is', cost
