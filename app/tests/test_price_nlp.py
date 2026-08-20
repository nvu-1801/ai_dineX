import unittest
from app.services.rag_service import extract_price_info, extract_food_query

class TestPriceNLP(unittest.TestCase):
    def test_extract_price_info_under_50k(self):
        min_p, max_p, query = extract_price_info("món dưới 50k")
        self.assertEqual(max_p, 50000.0)

    def test_extract_price_info_under_100k(self):
        min_p, max_p, query = extract_price_info("cơm tấm dưới 100k")
        self.assertEqual(max_p, 100000.0)

    def test_extract_price_info_under_200k(self):
        min_p, max_p, query = extract_price_info("trà sữa dưới 200k")
        self.assertEqual(max_p, 200000.0)

    def test_extract_price_info_under_300k(self):
        min_p, max_p, query = extract_price_info("món lẩu dưới 300k")
        self.assertEqual(max_p, 300000.0)

    def test_extract_price_info_under_400k(self):
        min_p, max_p, query = extract_price_info("combo ăn gia đình dưới 400k")
        self.assertEqual(max_p, 400000.0)

    def test_extract_price_info_under_500k(self):
        min_p, max_p, query = extract_price_info("bàn tiệc hải sản dưới 500k")
        self.assertEqual(max_p, 500000.0)

    def test_extract_price_info_range(self):
        min_p, max_p, query = extract_price_info("cho 2 tô phở bò tầm 45k đến 50k nhé")
        self.assertEqual(min_p, 45000.0)
        self.assertEqual(max_p, 50000.0)

    def test_extract_food_query_conversational(self):
        q = extract_food_query("Tôi muốn ăn 2 phần bún chả nhé ạ")
        self.assertIn("bún chả", q)

if __name__ == '__main__':
    unittest.main()
