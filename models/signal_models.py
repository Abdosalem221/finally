# app/models/signal_models.py

"""
نماذج البيانات للإشارات والتحقق منها.
يمكن استبدالها بنماذج Pydantic أو SQLAlchemy أو ما يناسب تطبيقك.
"""

from typing import Optional, Dict, Any
from datetime import datetime

class Signal:
    """
    نموذج بسيط لتمثيل إشارة تداول.
    (يمكن تخصيص هذه الفئة لتناسب متطلباتك الفعلية)
    """
    def __init__(self, symbol: str, timeframe: str, signal_type: str,
                 entry_price: float, stop_loss: float, take_profit: float,
                 timestamp: Optional[datetime] = None, **kwargs):
        self.symbol = symbol
        self.timeframe = timeframe
        self.signal_type = signal_type # مثل 'BUY', 'SELL'
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.timestamp = timestamp if timestamp is not None else datetime.utcnow()
        # إضافة أي حقول إضافية
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """تحويل النموذج إلى قاموس."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'signal_type': self.signal_type,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            # إضافة حقول إضافية إذا وجدت
            **{k: v for k, v in self.__dict__.items() if k not in ['symbol', 'timeframe', 'signal_type', 'entry_price', 'stop_loss', 'take_profit', 'timestamp']}
        }

class SignalVerification:
    """
    نموذج بسيط لتمثيل نتيجة التحقق من الإشارة أو أدائها.
    (يمكن تخصيص هذه الفئة لتناسب متطلباتك الفعلية)
    """
    def __init__(self, signal_id: str, verification_timestamp: datetime,
                 result: str, details: Optional[Dict[str, Any]] = None):
        self.signal_id = signal_id # معرف الإشارة التي يتم التحقق منها
        self.verification_timestamp = verification_timestamp
        self.result = result # مثل 'Success', 'Failure', 'Partial'
        self.details = details if details is not None else {} # تفاصيل إضافية عن التحقق

    def to_dict(self) -> Dict[str, Any]:
        """تحويل النموذج إلى قاموس."""
        return {
            'signal_id': self.signal_id,
            'verification_timestamp': self.verification_timestamp.isoformat(),
            'result': self.result,
            'details': self.details
        }

# مثال على كيفية استخدامه (يمكن حذفه إذا لم يكن ضروريا)
# if __name__ == "__main__":
#     sample_signal = Signal(
#         symbol="BTCUSDT",
#         timeframe="1h",
#         signal_type="BUY",
#         entry_price=30000.0,
#         stop_loss=29500.0,
#         take_profit=31000.0,
#         extra_info="Based on RSI divergence"
#     )
#     print(sample_signal.to_dict())

#     sample_verification = SignalVerification(
#         signal_id="abc123xyz",
#         verification_timestamp=datetime.utcnow(),
#         result="Success",
#         details={"reason": "Target hit"}
#     )
#     print(sample_verification.to_dict())