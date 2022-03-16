APPLE = 100
APPLE_NUM = 2
ORANGE = 150
ORANGE_NUM = 3
TAX = 1.1

class MulLayer():
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x 
        self.y = y
        out = self.x * self.y
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
        
    
class AddLayer():
    def __init__(self):
        pass
    
    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


if __name__ == "__main__":

    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()
    
    # forward
    apple_price = mul_apple_layer.forward(APPLE, APPLE_NUM)
    orange_price = mul_orange_layer.forward(ORANGE, ORANGE_NUM)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(all_price, TAX)
    
    # backward
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)
    
    print ("price: ", int(price))
    print ("dApple: ", dapple)
    print ("dApple_num: ", int(dapple_num))
    print ("dOrange: ", dorange)
    print ("dOrange_num: ", int(dorange_num))
    print ("dTax: ", dtax)
    