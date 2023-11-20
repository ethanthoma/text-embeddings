import unittest

from src.lib.tuple_mapper import tpm


class TestTupleMapper(unittest.TestCase):
    def setUp(self):
        self.string_val = 'one'
        self.number_val = 2
        self.array_val  = [3]

        def takes_3_and_returns_1(string, number, array):
            self.assertEqual(string, self.string_val)
            self.assertEqual(number, self.number_val)
            self.assertEqual(array, self.array_val)

            return string, number, array

        self.wrapped = tpm(takes_3_and_returns_1)

        self.expected_return = (
            self.string_val, 
            self.number_val, 
            self.array_val
        )


    def test_normal_call_case(self):
        actual_return = self.wrapped(
            self.string_val, 
            self.number_val, 
            self.array_val
        )

        self.assertEqual(actual_return, self.expected_return)


    def test_tuple_mapping_case(self):
        actual_return = self.wrapped(
            (
                self.string_val, 
                self.number_val, 
                self.array_val
            )
        )

        self.assertEqual(actual_return, self.expected_return)


    def test_curry_case(self):
        pass


if __name__ == '__main__':
    unittest.main()

