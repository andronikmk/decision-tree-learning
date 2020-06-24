import numpy as np

# training dataset
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




class DecisionTreeLearner:
    def __init__(self, examples, attributes, parent_examples, rows):
        self.examples = examples
        self.attributes = attributes
        self.parent_examples = parent_examples
        self.rows = training_data

    def class_counts(self):
        counts = {}
        for row in rows:
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts

    def gini(self):
        counts = class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl**2
        return impurity

    def info_gain(left, right, current_uncertainty):
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * gini(left) - (1 - p) * gini(right)
    
    def decision_tree_learning(self):
        if self.examples is None:
            pass

        # if all examples hae the same classifications then return classufications
        # attribes are empty return plurality values(examples)

        else:
            A = np.argmax()