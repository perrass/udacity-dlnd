from numpy import exp, dot, random, array
import pandas as pd


class XorPerceptron():

    def __init__(self):
        self.weights = ((0.0, -1.0), (0.0, -1.0),
                        (1.0, 1.0), (1.0, 1.0), (1.0, 1.0))
        self.bias = (0.0, 0.0, -2.0, -2.0, -1.0)
        self.activate_function = lambda x: int(x >= 0)
        self.linear_combination = lambda x, y1, y2, z: \
            x[0] * y1 + x[1] * y2 + z
        self.lc_value = 0.0
        self.output = 0.0

    def _output(self, test_input):
        lc_1_1 = self.linear_combination(
            self.weights[0], test_input[0], test_input[1], self.bias[0])
        lc_1_2 = self.linear_combination(
            self.weights[1], test_input[1], test_input[0], self.bias[1])
        output_1_1, output_1_2 = self.activate_function(lc_1_1), \
            self.activate_function(lc_1_2)
        lc_2_1 = self.linear_combination(
            self.weights[2], output_1_1, test_input[0], self.bias[2])
        lc_2_2 = self.linear_combination(
            self.weights[3], output_1_2, test_input[1], self.bias[3])
        output_2_1, output_2_2 = self.activate_function(lc_2_1), \
            self.activate_function(lc_2_2)
        lc_3 = self.linear_combination(
            self.weights[4], output_2_1, output_2_2, self.bias[4])
        self.lc_value = lc_3
        self.output = self.activate_function(lc_3)

if __name__ == "__main__":
    test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    correct_outputs = [False, True, True, False]
    outputs = []

    for test_input, correct_output in zip(test_inputs, correct_outputs):
        xor_perceptron = XorPerceptron()
        xor_perceptron._output(test_input)
        is_correct_string = 'Yes' if xor_perceptron.output == correct_output else 'No'
        outputs.append([test_input[0], test_input[1],
                        xor_perceptron.lc_value, xor_perceptron.output, is_correct_string])

    num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
    output_frame = pd.DataFrame(outputs, columns=['Input1', 'Input2',
                                                  'Linear Combination', 'Output', 'Is Correct'])
    if not num_wrong:
        print ('Nice! You got it all correct.\n')
    else:
        print ('You got {} wrong. Keep trying!\n'.format(num_wrong))
    print(output_frame.to_string(index=False))
