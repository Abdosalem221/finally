
class AdvancedPatternAnalyzer:
    def __init__(self):
        self.patterns = {
            'double_top': {'min_points': 5, 'confidence': 0.8},
            'head_shoulders': {'min_points': 7, 'confidence': 0.85},
            'triangle': {'min_points': 4, 'confidence': 0.75},
            'flag': {'min_points': 6, 'confidence': 0.7}
        }
        
    def analyze_patterns(self, price_data):
        """تحليل الأنماط السعرية المتقدمة"""
        detected_patterns = []
        for pattern_name, criteria in self.patterns.items():
            if self.detect_pattern(price_data, pattern_name, criteria):
                detected_patterns.append({
                    'name': pattern_name,
                    'confidence': criteria['confidence'],
                    'points': self.get_pattern_points(price_data, pattern_name)
                })
        return detected_patterns
