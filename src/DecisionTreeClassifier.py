training_data = [
    ['Yes', 'No','No','Yes','Some','$$$','No','Yes','French','0-10','Yes'],
    ['Yes', 'No', 'No', 'Yes', 'Full', '$', 'No', 'No', 'Thai', '30-60', 'No'],
    ['No', 'Yes', 'No', 'No', 'Some', '$', 'No', 'No', 'Burger', '0-10', 'Yes'],
    ['Yes', 'No', 'Yes', 'Yes', 'Full', '$', 'No', 'No', 'Thai', '10-30', 'Yes'],
    ['Yes', 'No', 'Yes', 'No', 'Full', '$$$', 'No', 'Yes', 'French', '>60', 'No'],
    ['No','Yes','No','Yes','Some','$$','Yes','Yes','Italian','0-10','Yes'],
    ['No', 'Yes', 'No', 'No', 'None', '$', 'Yes', 'No', 'Burger', '0-10', 'No'],
    ['No', 'No', 'No', 'Yes', 'Some', '$$', 'Yes', 'Yes', 'Thai', '0-10', 'Yes'],
    ['No', 'Yes', 'Yes', 'No', 'Full', '$', 'Yes', 'No', 'Burger', '>60', 'No'],
    ['Yes','Yes','Yes','Yes','Full','$$$','No','Yes','Italian','10-30','No'],
    ['No', 'No', 'No', 'No', 'None', '$', 'No', 'No', 'Thai', '0-10', 'No'],
    ['Yes','Yes','Yes','Yes','Full','$','No','No','Burger','30-60','Yes']
]

header = ["Alternate", "Bar", "Fri/Sat", "Hungry", "Patrons", "Price", "Raining", "Reservation" , "Type", "WaitEstimate", "WillWait"]

def unique_vals(rows, col):
    return set([row[col] for row in rows])

def class_counts(rows):
    counts = {}
    for row in rows:   
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

class Question:
    
    def __init__(self, column, value):
        self.column = column
        self.value = value
    
    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value
    
    def __repr__(self):        
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s" % (
        header[self.column], condition, str(self.value))

def partition(rows, question):
    
    true_rows, false_rows = [], []
    
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):
    best_gain = 0 
    best_question = None 
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1
    
    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question

class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)

class Decision_Node:

    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows):
    gain, question = find_best_split(rows)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision_Node(question, true_branch, false_branch)

def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return
    print(spacing + str(node.question))
    print(spacing + "---> True: ")
    print_tree(node.true_branch, spacing + "  ")
    print(spacing + "---> False: ")
    print_tree(node.false_branch, spacing + " ")


def predict(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return predict(row, node.true_branch)
    else:
        return predict(row, node.false_branch)

def fit(counts):

    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total*100)) + "%"
    return probs

my_tree = build_tree(training_data)

testing_data = [
    ['Yes', 'No','No','Yes','Some','$$$','No','Yes','French','0-10','Yes'],
    ['Yes', 'No', 'No', 'Yes', 'Full', '$', 'No', 'No', 'Thai', '30-60', 'No'],
    ['No', 'Yes', 'No', 'No', 'Some', '$', 'No', 'No', 'Burger', '0-10', 'Yes'],
    ['Yes', 'No', 'Yes', 'Yes', 'Full', '$', 'No', 'No', 'Thai', '10-30', 'Yes'],
]

for row in testing_data:
    print("Actual: %s Predicted: %s" % (row[-1], fit(predict(row, my_tree))))
