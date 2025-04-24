
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
        self.thresholds = {
            'accuracy': 0.85,
            'response_time': 0.5,
            'signal_quality': 0.8
        }
        
    def monitor_model_performance(self, model_name, predictions, actual):
        """مراقبة أداء النموذج في الوقت الفعلي"""
        metrics = self.calculate_metrics(predictions, actual)
        self.update_metrics(model_name, metrics)
        return self.check_performance(model_name)
        
    def calculate_metrics(self, predictions, actual):
        return {
            'accuracy': accuracy_score(actual, predictions),
            'precision': precision_score(actual, predictions),
            'recall': recall_score(actual, predictions)
        }
