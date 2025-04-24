
class RiskManager:
    def __init__(self):
        self.max_risk_per_trade = 0.02  # 2% per trade
        self.max_portfolio_risk = 0.06  # 6% total
        self.position_sizing_methods = ['fixed', 'volatility', 'kelly']
        
    def calculate_position_size(self, account_balance, risk_params):
        """حساب حجم العقد المناسب"""
        position_size = 0
        for method in self.position_sizing_methods:
            size = self.get_position_size(method, account_balance, risk_params)
            position_size = min(position_size + size, self.max_portfolio_risk * account_balance)
        return position_size
