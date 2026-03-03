import json
import unittest

from services.validator import parse_financial_number, validate_financial_data


class TestParseFinancialNumber(unittest.TestCase):
    def test_parse_vn_thousands(self) -> None:
        self.assertEqual(parse_financial_number("1.234.567"), 1234567.0)

    def test_parse_vn_decimal(self) -> None:
        self.assertEqual(parse_financial_number("1.234,56"), 1234.56)

    def test_parse_international_decimal(self) -> None:
        self.assertEqual(parse_financial_number("1,234.56"), 1234.56)

    def test_parse_accounting_negative(self) -> None:
        self.assertEqual(parse_financial_number("(1.234,56)"), -1234.56)

    def test_parse_with_currency_and_spaces(self) -> None:
        self.assertEqual(parse_financial_number(" VND 2 345 678 "), 2345678.0)

    def test_parse_invalid(self) -> None:
        self.assertIsNone(parse_financial_number("N/A"))


class TestValidateFinancialData(unittest.TestCase):
    def test_validation_pass(self) -> None:
        payload = {
            "total_assets": "3.000",
            "short_term_assets": "1.000",
            "long_term_assets": "2.000",
        }
        result = json.loads(validate_financial_data(json.dumps(payload)))
        self.assertNotIn("requires_human_review", result)

    def test_validation_fail_sets_flag(self) -> None:
        payload = {
            "total_assets": "2.900",
            "short_term_assets": "1.000",
            "long_term_assets": "2.000",
        }
        result = json.loads(validate_financial_data(json.dumps(payload)))
        self.assertTrue(result.get("requires_human_review"))

    def test_validation_with_vn_number_format(self) -> None:
        payload = {
            "total_assets": "1.234,56",
            "short_term_assets": "1.000,00",
            "long_term_assets": "234,56",
        }
        result = json.loads(validate_financial_data(json.dumps(payload)))
        self.assertNotIn("requires_human_review", result)

    def test_invalid_json_requires_review(self) -> None:
        result = json.loads(validate_financial_data("{not_json"))
        self.assertTrue(result.get("requires_human_review"))
        self.assertIn("raw", result)


if __name__ == "__main__":
    unittest.main()
