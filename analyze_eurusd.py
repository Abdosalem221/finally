
from app.services.market_data_service import MarketDataService
from app.services.analysis_service import AnalysisService
from app.services.ai_analysis import AIAnalysis
from flask import Flask
import json

app = Flask(__name__)

def analyze_eurusd():
    # تهيئة الخدمات
    market_service = MarketDataService(app)
    analysis_service = AnalysisService()
    ai_service = AIAnalysis()
    
    symbol = "EUR/USD"
    timeframe = "1m"
    
    # جلب بيانات السوق
    market_data = market_service.get_historical_data(symbol, timeframe, limit=100)
    
    if market_data is not None:
        # التحليل الفني
        technical_analysis = analysis_service.analyze_market(symbol, timeframe)
        
        # التحليل بالذكاء الاصطناعي
        ai_prediction = ai_service.predict(symbol, data=market_data, market_type='forex', period='1m')
        
        print("\n=== تحليل EUR/USD ===")
        print(f"السعر الحالي: {market_data['close'].iloc[-1]:.4f}")
        
        if technical_analysis['status'] == 'success':
            analysis = technical_analysis['analysis']
            
            print("\n-- تحليل الاتجاه --")
            print(f"الاتجاه قصير المدى: {analysis['trend']['ma']['short_term']}")
            print(f"الاتجاه متوسط المدى: {analysis['trend']['ma']['medium_term']}")
            print(f"قوة الاتجاه: {analysis['trend']['trend_strength']:.2f}")
            
            print("\n-- تحليل القوة --")
            print(f"مؤشر RSI: {analysis['strength']['rsi']['value']:.2f}")
            print(f"حالة RSI: {analysis['strength']['rsi']['strength']}")
            
            print("\n-- تحليل التقلب --")
            print(f"مستوى التقلب: {analysis['volatility']['volatility_level']}")
            
        if ai_prediction:
            print("\n-- توقعات الذكاء الاصطناعي --")
            print(f"الإشارة: {ai_prediction['signal']}")
            print(f"قوة الإشارة: {ai_prediction['strength']:.2f}")
            print(f"نسبة الثقة: {ai_prediction['confidence']:.2f}")
            
if __name__ == "__main__":
    analyze_eurusd()
