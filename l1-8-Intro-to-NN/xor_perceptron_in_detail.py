import numpy as np
import pandas as pd


class XorPerceptron(object):

    def __init__(self):
        """
        初始化神经元
        weights: 第一层有四个神经元位，两个是有权重的神经元，两个是Pass，
                 第二层两个神经元，第三层一个
        bias: weights对应的偏置
        activate_function: 激活函数，单位跃阶函数 (Heavisible step function)
        linear_combination: 神经元的计算方式
        """
        self.weights = ((0.0, -1.0), (0.0, -1.0),
                        (1.0, 1.0), (1.0, 1.0), (1.0, 1.0))
        self.bias = (0.0, 0.0, -2.0, -2.0, -1.0)
        self.activate_function = lambda x: int(x >= 0)
        # x: weights, y: input, z: bias
        self.linear_combination = lambda x, y1, y2, z: \
            x[0] * y1 + x[1] * y2 + z

    def output(self, test_input):
        """
        通过神经网络得到输出
        :param test_input: 输入
        :return: the value of linear_combination at the last layer, output
        """
        assert len(test_input) == 2
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

        lc_value = lc_3
        output = self.activate_function(lc_3)

        return lc_value, output


def run():
    test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    correct_outputs = [False, True, True, False]
    outputs = []

    for test_input, correct_output in zip(test_inputs, correct_outputs):
        xor_perceptron = XorPerceptron()
        lc_value, xor_output = xor_perceptron.output(test_input)
        is_correct_string = 'Yes' if xor_output == correct_output else 'No'
        outputs.append([test_input[0], test_input[1],
                        lc_value, xor_output, is_correct_string])

    num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
    output_frame = pd.DataFrame(outputs, columns=['Input1', 'Input2',
                                                  'Linear Combination', 'Output', 'Is Correct'])
    if not num_wrong:
        print('Nice! You got it all correct.\n')
    else:
        print('You got {} wrong. Keep trying!\n'.format(num_wrong))
    print(output_frame.to_string(index=False))


if __name__ == "__main__":
    run()
