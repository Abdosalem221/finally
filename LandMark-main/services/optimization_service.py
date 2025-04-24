"""
خدمة تحسين استراتيجيات التداول
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from app.services.data_service import DataService
from app.services.backtest_service import BacktestService
from app.utils.helpers import format_percentage, format_timestamp
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import asyncio

class OptimizationService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_service = DataService()
        self.backtest_service = BacktestService()
    
    def optimize_strategy(self, strategy: Dict,
                         symbol: str,
                         timeframe: str,
                         start_date: datetime,
                         end_date: datetime,
                         method: str = 'grid_search',
                         optimization_params: Optional[Dict] = None) -> Dict:
        """
        تحسين استراتيجية التداول
        
        Args:
            strategy: استراتيجية التداول
            symbol: رمز العملة
            timeframe: الإطار الزمني
            start_date: تاريخ البداية
            end_date: تاريخ النهاية
            method: طريقة التحسين
            optimization_params: معلمات التحسين
            
        Returns:
            Dict: نتائج التحسين
        """
        try:
            # جلب بيانات السوق
            market_data = self.data_service.fetch_market_data(
                symbol,
                timeframe,
                start_date,
                end_date
            )
            
            if market_data['status'] == 'error':
                return {'status': 'error', 'message': 'فشل في جلب بيانات السوق'}
            
            # تحويل البيانات إلى DataFrame
            df = pd.DataFrame(market_data['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # اختيار طريقة التحسين
            if method == 'grid_search':
                best_params, best_score = self._grid_search(
                    strategy,
                    df,
                    optimization_params
                )
            elif method == 'genetic_algorithm':
                best_params, best_score = self._genetic_algorithm(
                    strategy,
                    df,
                    optimization_params
                )
            elif method == 'bayesian_optimization':
                best_params, best_score = self._bayesian_optimization(
                    strategy,
                    df,
                    optimization_params
                )
            else:
                return {'status': 'error', 'message': 'طريقة تحسين غير صالحة'}
            
            return {
                'status': 'success',
                'best_params': best_params,
                'best_score': best_score,
                'optimization_method': method,
                'optimization_date': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing strategy: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _grid_search(self, strategy: Dict,
                    data: pd.DataFrame,
                    params: Optional[Dict] = None) -> Tuple[Dict, float]:
        """
        تحسين باستخدام البحث الشبكي
        
        Args:
            strategy: استراتيجية التداول
            data: بيانات السوق
            params: معلمات التحسين
            
        Returns:
            Tuple[Dict, float]: أفضل المعلمات والنتيجة
        """
        try:
            if not params:
                params = {
                    'sma_period': range(10, 51, 10),
                    'rsi_period': range(10, 31, 5),
                    'rsi_oversold': range(20, 41, 5),
                    'rsi_overbought': range(60, 81, 5)
                }
            
            best_score = float('-inf')
            best_params = {}
            
            # توليد جميع التركيبات الممكنة
            param_combinations = self._generate_param_combinations(params)
            
            for combination in param_combinations:
                # تطبيق الاستراتيجية
                strategy_copy = strategy.copy()
                strategy_copy['params'].update(combination)
                
                # اختبار الاستراتيجية
                result = self.backtest_service.run_backtest(
                    strategy_copy,
                    data
                )
                
                if result['status'] == 'error':
                    continue
                
                # حساب درجة الأداء
                score = self._calculate_performance_score(result)
                
                if score > best_score:
                    best_score = score
                    best_params = combination
            
            return best_params, best_score
            
        except Exception as e:
            self.logger.error(f"Error in grid search: {str(e)}")
            raise
    
    def _genetic_algorithm(self, strategy: Dict,
                          data: pd.DataFrame,
                          params: Optional[Dict] = None) -> Tuple[Dict, float]:
        """
        تحسين باستخدام الخوارزمية الجينية
        
        Args:
            strategy: استراتيجية التداول
            data: بيانات السوق
            params: معلمات التحسين
            
        Returns:
            Tuple[Dict, float]: أفضل المعلمات والنتيجة
        """
        try:
            if not params:
                params = {
                    'population_size': 50,
                    'generations': 20,
                    'mutation_rate': 0.1,
                    'param_ranges': {
                        'sma_period': (10, 50),
                        'rsi_period': (10, 30),
                        'rsi_oversold': (20, 40),
                        'rsi_overbought': (60, 80)
                    }
                }
            
            population = self._initialize_population(
                params['population_size'],
                params['param_ranges']
            )
            
            best_score = float('-inf')
            best_params = {}
            
            for generation in range(params['generations']):
                scores = []
                
                for individual in population:
                    # تطبيق الاستراتيجية
                    strategy_copy = strategy.copy()
                    strategy_copy['params'].update(individual)
                    
                    # اختبار الاستراتيجية
                    result = self.backtest_service.run_backtest(
                        strategy_copy,
                        data
                    )
                    
                    if result['status'] == 'error':
                        scores.append(float('-inf'))
                        continue
                    
                    # حساب درجة الأداء
                    score = self._calculate_performance_score(result)
                    scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_params = individual
                
                # اختيار الأفضل
                selected = self._select_best_individuals(
                    population,
                    scores,
                    params['population_size'] // 2
                )
                
                # توليد جيل جديد
                population = self._generate_new_generation(
                    selected,
                    params['mutation_rate'],
                    params['param_ranges']
                )
            
            return best_params, best_score
            
        except Exception as e:
            self.logger.error(f"Error in genetic algorithm: {str(e)}")
            raise
    
    def _bayesian_optimization(self, strategy: Dict,
                             data: pd.DataFrame,
                             params: Optional[Dict] = None) -> Tuple[Dict, float]:
        """
        تحسين باستخدام التحسين البايزي
        
        Args:
            strategy: استراتيجية التداول
            data: بيانات السوق
            params: معلمات التحسين
            
        Returns:
            Tuple[Dict, float]: أفضل المعلمات والنتيجة
        """
        try:
            if not params:
                params = {
                    'n_initial_points': 10,
                    'n_iterations': 20,
                    'param_ranges': {
                        'sma_period': (10, 50),
                        'rsi_period': (10, 30),
                        'rsi_oversold': (20, 40),
                        'rsi_overbought': (60, 80)
                    }
                }
            
            # تهيئة النقاط الأولية
            points = self._generate_random_points(
                params['n_initial_points'],
                params['param_ranges']
            )
            
            best_score = float('-inf')
            best_params = {}
            
            for point in points:
                # تطبيق الاستراتيجية
                strategy_copy = strategy.copy()
                strategy_copy['params'].update(point)
                
                # اختبار الاستراتيجية
                result = self.backtest_service.run_backtest(
                    strategy_copy,
                    data
                )
                
                if result['status'] == 'error':
                    continue
                
                # حساب درجة الأداء
                score = self._calculate_performance_score(result)
                
                if score > best_score:
                    best_score = score
                    best_params = point
            
            # التحسين التكراري
            for _ in range(params['n_iterations']):
                # اختيار النقطة التالية
                next_point = self._select_next_point(
                    points,
                    params['param_ranges']
                )
                
                # تطبيق الاستراتيجية
                strategy_copy = strategy.copy()
                strategy_copy['params'].update(next_point)
                
                # اختبار الاستراتيجية
                result = self.backtest_service.run_backtest(
                    strategy_copy,
                    data
                )
                
                if result['status'] == 'error':
                    continue
                
                # حساب درجة الأداء
                score = self._calculate_performance_score(result)
                
                if score > best_score:
                    best_score = score
                    best_params = next_point
                
                points.append(next_point)
            
            return best_params, best_score
            
        except Exception as e:
            self.logger.error(f"Error in Bayesian optimization: {str(e)}")
            raise
    
    def _generate_param_combinations(self, params: Dict) -> List[Dict]:
        """
        توليد تركيبات المعلمات
        
        Args:
            params: معلمات التحسين
            
        Returns:
            List[Dict]: قائمة تركيبات المعلمات
        """
        try:
            from itertools import product
            
            keys = params.keys()
            values = params.values()
            
            combinations = []
            for combination in product(*values):
                combinations.append(dict(zip(keys, combination)))
            
            return combinations
            
        except Exception as e:
            self.logger.error(f"Error generating parameter combinations: {str(e)}")
            raise
    
    def _calculate_performance_score(self, result: Dict) -> float:
        """
        حساب درجة الأداء
        
        Args:
            result: نتائج الاختبار
            
        Returns:
            float: درجة الأداء
        """
        try:
            # حساب الدرجة بناءً على معايير متعددة
            profit_factor = result.get('profit_factor', 0)
            win_rate = result.get('win_rate', 0)
            sharpe_ratio = result.get('sharpe_ratio', 0)
            max_drawdown = result.get('max_drawdown', 0)
            
            # وزن المعايير
            score = (
                profit_factor * 0.3 +
                win_rate * 0.3 +
                sharpe_ratio * 0.2 +
                (1 - max_drawdown) * 0.2
            )
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating performance score: {str(e)}")
            return float('-inf')
    
    def _initialize_population(self, size: int,
                             param_ranges: Dict) -> List[Dict]:
        """
        تهيئة المجتمع الأولي
        
        Args:
            size: حجم المجتمع
            param_ranges: نطاقات المعلمات
            
        Returns:
            List[Dict]: المجتمع الأولي
        """
        try:
            population = []
            
            for _ in range(size):
                individual = {}
                for param, (min_val, max_val) in param_ranges.items():
                    individual[param] = np.random.randint(min_val, max_val + 1)
                
                population.append(individual)
            
            return population
            
        except Exception as e:
            self.logger.error(f"Error initializing population: {str(e)}")
            raise
    
    def _select_best_individuals(self, population: List[Dict],
                               scores: List[float],
                               n: int) -> List[Dict]:
        """
        اختيار أفضل الأفراد
        
        Args:
            population: المجتمع
            scores: درجات الأفراد
            n: عدد الأفراد المطلوب اختيارهم
            
        Returns:
            List[Dict]: أفضل الأفراد
        """
        try:
            # دمج المجتمع مع الدرجات
            combined = list(zip(population, scores))
            
            # ترتيب تنازلي حسب الدرجة
            combined.sort(key=lambda x: x[1], reverse=True)
            
            # اختيار أفضل n فرد
            return [individual for individual, _ in combined[:n]]
            
        except Exception as e:
            self.logger.error(f"Error selecting best individuals: {str(e)}")
            raise
    
    def _generate_new_generation(self, selected: List[Dict],
                               mutation_rate: float,
                               param_ranges: Dict) -> List[Dict]:
        """
        توليد جيل جديد
        
        Args:
            selected: الأفراد المختارة
            mutation_rate: معدل الطفرة
            param_ranges: نطاقات المعلمات
            
        Returns:
            List[Dict]: الجيل الجديد
        """
        try:
            new_generation = []
            
            while len(new_generation) < len(selected) * 2:
                # اختيار والدين عشوائيين
                parent1, parent2 = np.random.choice(selected, 2, replace=False)
                
                # التهجين
                child = {}
                for param in parent1.keys():
                    if np.random.random() < 0.5:
                        child[param] = parent1[param]
                    else:
                        child[param] = parent2[param]
                
                # الطفرة
                if np.random.random() < mutation_rate:
                    param = np.random.choice(list(param_ranges.keys()))
                    min_val, max_val = param_ranges[param]
                    child[param] = np.random.randint(min_val, max_val + 1)
                
                new_generation.append(child)
            
            return new_generation
            
        except Exception as e:
            self.logger.error(f"Error generating new generation: {str(e)}")
            raise
    
    def _generate_random_points(self, n: int,
                              param_ranges: Dict) -> List[Dict]:
        """
        توليد نقاط عشوائية
        
        Args:
            n: عدد النقاط
            param_ranges: نطاقات المعلمات
            
        Returns:
            List[Dict]: النقاط العشوائية
        """
        try:
            points = []
            
            for _ in range(n):
                point = {}
                for param, (min_val, max_val) in param_ranges.items():
                    point[param] = np.random.randint(min_val, max_val + 1)
                
                points.append(point)
            
            return points
            
        except Exception as e:
            self.logger.error(f"Error generating random points: {str(e)}")
            raise
    
    def _select_next_point(self, points: List[Dict],
                          param_ranges: Dict) -> Dict:
        """
        اختيار النقطة التالية
        
        Args:
            points: النقاط الحالية
            param_ranges: نطاقات المعلمات
            
        Returns:
            Dict: النقطة التالية
        """
        try:
            # حساب متوسط المسافات
            distances = []
            for i in range(len(points)):
                dist = 0
                for j in range(len(points)):
                    if i != j:
                        for param in points[i].keys():
                            dist += abs(points[i][param] - points[j][param])
                distances.append(dist)
            
            # اختيار النقطة الأبعد
            next_point = points[np.argmax(distances)]
            
            # إضافة تغيير عشوائي
            for param in next_point.keys():
                if np.random.random() < 0.3:
                    min_val, max_val = param_ranges[param]
                    next_point[param] = np.random.randint(min_val, max_val + 1)
            
            return next_point
            
        except Exception as e:
            self.logger.error(f"Error selecting next point: {str(e)}")
            raise 


class EnhancedOptimizationService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.model = self._build_deep_model()
        self.optimization_queue = asyncio.Queue()
        self.is_optimizing = False

    def _build_deep_model(self):
        """بناء نموذج التعلم العميق"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    async def optimize_strategy(self, strategy: Dict, data: pd.DataFrame) -> Dict:
        """تحسين استراتيجية باستخدام التعلم العميق"""
        try:
            # تحضير البيانات
            X = self._prepare_features(data)
            y = self._prepare_targets(data)

            # تدريب النموذج
            history = self.model.fit(
                X, y,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )

            # تحسين المعلمات
            optimized_params = await self._optimize_parameters(strategy, X)

            return {
                'status': 'success',
                'optimized_parameters': optimized_params,
                'performance_metrics': {
                    'accuracy': history.history['accuracy'][-1],
                    'loss': history.history['loss'][-1]
                }
            }

        except Exception as e:
            self.logger.error(f"Error in strategy optimization: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    async def _optimize_parameters(self, strategy: Dict, X: np.ndarray) -> Dict:
        """تحسين معلمات الاستراتيجية"""
        predictions = self.model.predict(X)

        # تحسين المعلمات بناءً على التنبؤات
        optimized = {
            'entry_threshold': np.percentile(predictions, 75),
            'exit_threshold': np.percentile(predictions, 25),
            'stop_loss': self._calculate_optimal_stop_loss(predictions),
            'take_profit': self._calculate_optimal_take_profit(predictions)
        }

        return optimized

    def _calculate_optimal_stop_loss(self, predictions: np.ndarray) -> float:
        """حساب مستوى وقف الخسارة الأمثل"""
        return float(np.percentile(predictions, 10))

    def _calculate_optimal_take_profit(self, predictions: np.ndarray) -> float:
        """حساب مستوى جني الأرباح الأمثل"""
        return float(np.percentile(predictions, 90))

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """تحضير ميزات النموذج"""
        # هنا يجب إضافة منطق لتحضير الميزات من بيانات السوق
        # مثال: استخدام مؤشرات فنية مثل SMA, RSI, MACD, etc.
        features = data[['open', 'high', 'low', 'close', 'volume']]
        scaled_features = self.scaler.fit_transform(features)
        return scaled_features

    def _prepare_targets(self, data: pd.DataFrame) -> np.ndarray:
        """تحضير أهداف النموذج"""
        # هنا يجب إضافة منطق لتحضير الأهداف من بيانات السوق
        # مثال: 1 إذا كان السعر سيرتفع, 0 إذا كان سيرتد
        data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
        targets = data['target'].values[:-1]  # Remove last element due to shift
        return targets