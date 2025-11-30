import numpy as np
import random
import time
import matplotlib.pyplot as plt


# =================================================================
# 1. Цільова функція (Розенброка, d=10)
# =================================================================

def rosenbrock_function(x):
    """
    Функція Долина Розенброка для d=10. Глобальний мінімум f(x)=0 при x_i=1.0.
    """
    d = len(x)
    sum_val = 0
    for i in range(d - 1):
        # f(x)=sum_{i=1}^{d-1}[100(x_{i+1}-x_{i}^{2})^{2}+(1-x_{i})^{2}]
        sum_val += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return sum_val


# =================================================================
# 2. Структура даних для Джерела Їжі (Food Source)
# =================================================================

class FoodSource:
    def __init__(self, n_dim, bounds_low, bounds_high, func):
        self.n_dim = n_dim
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.func = func

        # Випадкова ініціалізація позиції
        self.position = np.random.uniform(low=bounds_low, high=bounds_high, size=n_dim)
        self.value = func(self.position)
        self.fitness = self.calculate_fitness(self.value)
        self.trials = 0  # Лічильник невдалих спроб покращення

    @staticmethod
    def calculate_fitness(value):
        """
        Обчислення пристосованості (Fitness) для мінімізації.
        Чим менше value (значення функції), тим вище fitness.
        """
        if value >= 0:
            return 1.0 / (1.0 + value)
        # Для від'ємних значень (хоча Розенброк >= 0)
        return 1.0 + abs(value)


# =================================================================
# 3. Алгоритм ABC
# =================================================================

class ABC:
    def __init__(self, func, n_dim, bounds, n_employed, n_onlooker, n_iterations, limit):

        # Параметри задачі
        self.func = func
        self.n_dim = n_dim
        self.bounds_low, self.bounds_high = bounds

        # Гіперпараметри ABC
        self.n_employed = n_employed  # Кількість джерел їжі (Food Sources) = Employed Bees
        self.n_onlooker = n_onlooker  # Кількість Onlooker Bees
        self.n_iterations = n_iterations
        self.limit = limit  # Поріг для перетворення на Scout Bee

        # Колонія
        self.sources = []

        # Глобально найкращий розв'язок
        self.gbest_position = None
        self.gbest_value = float('inf')

        self.convergence_history = []

        # Ініціалізація
        self._initialize_sources()

    def _initialize_sources(self):
        """Ініціалізація SN джерел їжі."""
        self.sources = [FoodSource(self.n_dim, self.bounds_low, self.bounds_high, self.func)
                        for _ in range(self.n_employed)]

        # Оновлення глобального найкращого значення
        self._update_gbest(self.sources)

    def _update_gbest(self, sources):
        """Перевіряє і оновлює глобально найкращий розв'язок."""
        for source in sources:
            if source.value < self.gbest_value:
                self.gbest_value = source.value
                self.gbest_position = np.copy(source.position)

    @staticmethod
    def _local_search(current_source, all_sources):
        """
        Виконує локальний пошук (Фаза Employed/Onlooker).
        v_ij = x_ij + phi_ij * (x_ij - x_kj)
        """
        n_dim = current_source.n_dim

        # 1. Випадково обираємо інше джерело k (k != i)
        k_idx = random.choice([i for i in range(len(all_sources)) if all_sources[i] != current_source])
        k_source = all_sources[k_idx]

        # 2. Створюємо нову позицію v
        new_position = np.copy(current_source.position)

        # Обираємо випадкову розмірність j для зміни
        j = np.random.randint(0, n_dim)

        # Генеруємо phi_ij з U(-1, 1)
        phi_ij = np.random.uniform(-1, 1)

        # Застосовуємо формулу оновлення до j-тої розмірності
        new_position[j] = current_source.position[j] + phi_ij * (current_source.position[j] - k_source.position[j])

        # Обробка меж: Відсікання (clamping)
        new_position = np.clip(new_position, current_source.bounds_low, current_source.bounds_high)

        # 3. Обчислюємо значення
        new_value = current_source.func(new_position)

        # 4. Жадібний вибір
        if new_value < current_source.value:
            current_source.position = new_position
            current_source.value = new_value
            current_source.fitness = FoodSource.calculate_fitness(new_value)
            current_source.trials = 0  # Скидання лічильника
        else:
            current_source.trials += 1

    def _calculate_probabilities(self, method='classic', alpha=5.0):
        """
        Обчислює ймовірності вибору джерел для Onlooker Bees.
        method='classic': p_i = fit_i / sum(fit_j)
        method='aggressive': p_i = exp(-alpha * f_i) / sum(exp(-alpha * f_j))
        """
        values = np.array([s.value for s in self.sources])

        if method == 'classic':
            # 1. Перетворюємо values на fitness (це вже зроблено в FoodSource)
            fitnesses = np.array([s.fitness for s in self.sources])
            sum_fitness = np.sum(fitnesses)

            if sum_fitness == 0:
                return np.ones(len(self.sources)) / len(self.sources)

            # Класична пропорція
            probabilities = fitnesses / sum_fitness

        elif method == 'aggressive':
            # 2. Softmax-стиль (агресивний)
            # Мінімальне значення потрібне для чисельної стабільності
            stable_values = values - np.min(values)

            # exp(-alpha * f_i)
            numerator = np.exp(-alpha * stable_values)
            denominator = np.sum(numerator)

            if denominator == 0:
                return np.ones(len(self.sources)) / len(self.sources)

            probabilities = numerator / denominator

        else:
            raise ValueError("Невідомий метод обчислення ймовірностей.")

        return probabilities

    def solve(self, prob_method='classic', alpha=5.0):

        # Головний цикл
        for t in range(self.n_iterations):

            # ====================================
            # 1. Фаза Employed Bees
            # ====================================
            for source in self.sources:
                self._local_search(source, self.sources)

            # Оновлення GBest після фази Employed
            self._update_gbest(self.sources)

            # ====================================
            # 2. Фаза Onlooker Bees
            # ====================================

            # Обчислення ймовірностей
            probabilities = self._calculate_probabilities(prob_method, alpha)

            # Розподіл завдань для N_onlooker бджіл
            # використовуємо np.random.choice для вибору джерел
            chosen_indices = np.random.choice(
                a=len(self.sources),
                size=self.n_onlooker,
                p=probabilities,
                replace=True  # Заміна дозволяє обирати одне джерело кілька разів
            )

            for index in chosen_indices:
                self._local_search(self.sources[index], self.sources)

            # Оновлення GBest після фази Onlooker
            self._update_gbest(self.sources)

            # ====================================
            # 3. Фаза Scout Bees
            # ====================================

            for i, source in enumerate(self.sources):
                if source.trials >= self.limit:
                    # Створення нового джерела (Scout)
                    new_source = FoodSource(self.n_dim, self.bounds_low, self.bounds_high, self.func)
                    self.sources[i] = new_source  # Заміна виснаженого джерела новим

                    # Оновлення GBest, якщо нове джерело краще
                    self._update_gbest([new_source])

            # Збереження історії збіжності
            self.convergence_history.append(self.gbest_value)

        return self.gbest_position, self.gbest_value, self.convergence_history


# =================================================================
# 4. Блок для запуску експериментів (MAIN)
# =================================================================

if __name__ == "__main__":

    # --- Фіксовані параметри задачі та алгоритму ---
    DIMENSION = 10
    BOUNDS = [-2.048, 2.048]
    N_RUNS = 10
    N_ITERATIONS = 1000  # Критерій зупинки
    BASE_SN = 40  # Базова кількість джерел (Employed Bees)
    LIMIT = 10 * DIMENSION  # 10*d = 100

    # --- Експерименти 1: Вплив частки Onlooker Bees (використовуємо classic prob) ---
    onlooker_ratios = {
        "E1.1 (25%)": 13,  # N_E=40, N_O=13
        "E1.2 (50%)": 40,  # N_E=40, N_O=40
        "E1.3 (75%)": 120  # N_E=40, N_O=120
    }

    # --- Експерименти 2: Агресивний вибір джерел (фіксуємо N_E=N_O=40) ---
    prob_methods = {
        "E2.1 (Classic)": ('classic', 0.0),  # Базовий ABC
        "E2.2 (Aggressive, a=5)": ('aggressive', 5.0),
        "E2.3 (Very Aggressive, a=10)": ('aggressive', 10.0)
    }

    all_experiment_results = {}

    print("--- Розпочато експерименти ЛР №3: ABC (Розенброк d=10) ---")

    # -------------------------------------------------------------
    # ЕКСПЕРИМЕНТ 1: Частка Onlooker Bees
    # -------------------------------------------------------------
    print("\n--- Експеримент 1: Вплив N_Onlooker (Classic Probability) ---")
    for name, n_onlooker in onlooker_ratios.items():
        print(f"Запуск: {name} (N_E={BASE_SN}, N_O={n_onlooker})")

        run_results = []
        run_histories = []
        start_time = time.time()

        for i in range(N_RUNS):
            np.random.seed(i)
            random.seed(i)

            abc = ABC(rosenbrock_function, DIMENSION, BOUNDS,
                      n_employed=BASE_SN, n_onlooker=n_onlooker,
                      n_iterations=N_ITERATIONS, limit=LIMIT)

            _, val, hist = abc.solve(prob_method='classic')
            run_results.append(val)
            run_histories.append(hist)

        end_time = time.time()

        # Зберігання результатів
        all_experiment_results[name] = {
            'best': np.min(run_results),
            'mean': np.mean(run_results),
            'std': np.std(run_results),
            'time': end_time - start_time,
            'histories': run_histories
        }

        print(
            f"  Best: {all_experiment_results[name]['best']:.4e}, Mean±Std: {all_experiment_results[name]['mean']:.4e} ± {all_experiment_results[name]['std']:.4e}")

    # -------------------------------------------------------------
    # ЕКСПЕРИМЕНТ 2: Агресивний вибір джерел
    # -------------------------------------------------------------
    print(f"\n--- Експеримент 2: Вплив формули ймовірності (N_E=N_O={BASE_SN}) ---")
    for name, (method, alpha) in prob_methods.items():
        print(f"Запуск: {name} (Alpha={alpha})")

        run_results = []
        run_histories = []
        start_time = time.time()

        # Фіксуємо N_Employed = N_Onlooker для цього експерименту
        N_E2 = BASE_SN
        N_O2 = BASE_SN

        for i in range(N_RUNS):
            np.random.seed(i)
            random.seed(i)

            abc = ABC(rosenbrock_function, DIMENSION, BOUNDS,
                      n_employed=N_E2, n_onlooker=N_O2,
                      n_iterations=N_ITERATIONS, limit=LIMIT)

            _, val, hist = abc.solve(prob_method=method, alpha=alpha)
            run_results.append(val)
            run_histories.append(hist)

        end_time = time.time()

        # Зберігання результатів
        all_experiment_results[name] = {
            'best': np.min(run_results),
            'mean': np.mean(run_results),
            'std': np.std(run_results),
            'time': end_time - start_time,
            'histories': run_histories
        }

        print(
            f"  Best: {all_experiment_results[name]['best']:.4e}, Mean±Std: {all_experiment_results[name]['mean']:.4e} ± {all_experiment_results[name]['std']:.4e}")

    # =================================================================
    # 5. Візуалізація результатів (Об'єднання всіх історій)
    # =================================================================

    plt.figure(figsize=(14, 8))

    print("\n--- Побудова графіка збіжності (середнє по 10 прогонах) ---")

    # Побудова графіка для кожного експерименту
    for name, data in all_experiment_results.items():
        avg_history = np.mean(np.array(data['histories']), axis=0)
        plt.semilogy(avg_history, label=f"ABC: {name}")

    # Додавання результатів gbest з ЛР2 для порівняння
    # Використовуємо середню історію з ЛР2 (gbest_avg_history), яку виводили раніше
    # plt.semilogy(gbest_avg_history, label="PSO: gbest (ЛР2)", color='black', linestyle=':')

    plt.title(f"Порівняння збіжності ABC для різних конфігурацій (Розенброк, d={DIMENSION})")
    plt.xlabel("Ітерація")
    plt.ylabel("Найкраще значення функції (log scale)")
    plt.legend()
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.ylim(bottom=1e-1)

    print("\nГрафік збіжності відображено.")
    plt.show()