from vsa_codebook_decoder import utils


class TestProduct:
    def test_product(self):
        assert 3 * 4 * 5 == utils.product([3, 4, 5])

class TestIOU:
    def test_iou_pytorch(self):
        assert True
