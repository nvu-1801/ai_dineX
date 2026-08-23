import unittest
from app.services.rag_service import normalize_vietnamese_food_typos, extract_price_info, extract_food_query
from app.services.chat_service import detect_out_of_domain
from app.routers.ai import UserLocationSchema
from pydantic import ValidationError


class TestAIFeatures(unittest.TestCase):
    def test_teencode_normalization(self):
        self.assertEqual(normalize_vietnamese_food_typos("1 bnh mi"), "1 bánh mì")
        self.assertEqual(normalize_vietnamese_food_typos("uong cf ko"), "uong cà phê không")
        self.assertEqual(normalize_vietnamese_food_typos("them 1 phan bún chỏ"), "them 1 phan bún chả")
        self.assertEqual(normalize_vietnamese_food_typos("uong bac xiu nha"), "uong bạc sỉu nha")

    def test_out_of_domain_guardrail(self):
        # Math queries should be blocked
        is_ood, reason = detect_out_of_domain("giải phương trình bậc 2: x^2 - 4x + 4 = 0")
        self.assertTrue(is_ood)

        # Programming queries should be blocked
        is_ood, reason = detect_out_of_domain("viết code python kết nối postgres")
        self.assertTrue(is_ood)

        # Food queries should NOT be blocked
        is_ood, reason = detect_out_of_domain("quán có bán phở bò không ạ")
        self.assertFalse(is_ood)

        # Substring safety check: 'văn' should not trigger 'ăn' whitelist when message is OOD translation
        is_ood, reason = detect_out_of_domain("dịch đoạn văn này sang tiếng anh")
        self.assertTrue(is_ood)

    def test_user_location_schema(self):
        # Both valid
        loc = UserLocationSchema(latitude=10.776, longitude=106.700)
        self.assertEqual(loc.latitude, 10.776)
        self.assertEqual(loc.longitude, 106.700)

        # Neither provided (valid)
        loc_empty = UserLocationSchema()
        self.assertIsNone(loc_empty.latitude)
        self.assertIsNone(loc_empty.longitude)

        # Partial latitude only (should fail)
        with self.assertRaises(ValidationError):
            UserLocationSchema(latitude=10.776)

        # Partial longitude only (should fail)
        with self.assertRaises(ValidationError):
            UserLocationSchema(longitude=106.700)

        # Out of bounds latitude
        with self.assertRaises(ValidationError):
            UserLocationSchema(latitude=95.0, longitude=106.0)


if __name__ == '__main__':
    unittest.main()
