import enum
import math
import random
import unittest
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, NewType
from operator import itemgetter


class PolicyType(enum.Enum):
    LFU = 'LFU'
    LRU = 'LRU'


CacheKey = NewType("CacheKey", str)


class CachePolicy:
    def __init__(self, budget=0):
        assert budget > 0
        self.budget = budget
        self.hits = 0
        self.accesses = 0

    def access(self, key: CacheKey) -> Optional[CacheKey]:
        pass

    @property
    def hitrate(self):
        return float(self.hits) * 100 / self.accesses


class LFU(CachePolicy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.frequency_counter = Counter()

    def __contains__(self, item):
        return self.frequency_counter[item] > 0

    def access(self, key: CacheKey) -> Optional[CacheKey]:
        self.accesses += 1
        if key in self:
            self.hits += 1

        self.frequency_counter[key] += 1
        if len(self.frequency_counter) > self.budget:
            # need to evict something
            least_frequent_key, _ = min(self.frequency_counter.items(), key=itemgetter(1))
            del self.frequency_counter[least_frequent_key]
            return least_frequent_key


class LRU(CachePolicy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recency_list = []

    def __contains__(self, item):
        return item in self.recency_list

    def access(self, key: CacheKey) -> Optional[CacheKey]:
        self.accesses += 1
        if key in self:
            self.hits += 1

        try:
            self.recency_list.remove(key)
        except ValueError:
            pass

        self.recency_list.insert(0, key)

        if len(self.recency_list) > self.budget:
            return self.recency_list.pop()


class LCR(CachePolicy):
    def __init__(self, budget=0):
        super().__init__(budget)
        self.history_lfu = []
        self.history_lru = []
        self.lfu_policy = LFU(budget=budget)
        self.lru_policy = LRU(budget=budget)
        self.w_lfu = 0.5
        self.w_lru = 0.5

    def __contains__(self, key):
        return key in self.lfu_policy or key in self.lru_policy

    def update_weight(self, key):
        learning_rate = 0.45
        # discount_rate = 0.005 ** (1 / self.budget)
        # r = 1  # update later to account for time in cache history
        if key in self.history_lru:
            self.w_lfu = self.w_lfu * math.exp(learning_rate)

        elif key in self.history_lfu:
            self.w_lru = self.w_lru * math.exp(learning_rate)

        self.w_lru = self.w_lru / (self.w_lru + self.w_lfu)
        self.w_lfu = 1 - self.w_lru

    def select_policy_and_history(self):
        if self.w_lru > self.w_lfu:
            return self.lru_policy, self.history_lru
        elif self.w_lfu > self.w_lru:
            return self.lfu_policy, self.history_lfu
        else:
            return random.choice([(self.lru_policy, self.history_lru), (self.lfu_policy, self.history_lfu)])

    def access(self, key: CacheKey) -> Optional[CacheKey]:
        self.accesses += 1

        if key in self:
            # hit
            self.hits += 1

            if key in self.lfu_policy:
                return self.lfu_policy.access(key)
            if key in self.lru_policy:
                return self.lru_policy.access(key)

        else:
            self.update_weight(key)
            policy, history = self.select_policy_and_history()
            maybe_evicted = policy.access(key)
            if maybe_evicted:
                if len(history) >= self.budget:
                    history.pop()
                history.insert(0, maybe_evicted)


class LCRTest(unittest.TestCase):

    def setUp(self) -> None:
        self.test_vector = [('a', None), ('b', None), ('a', None), ('c', 'b')]

    def test_lfu_lru(self):
        lfu = LFU(budget=2)
        lru = LRU(budget=2)

        for key, expected in self.test_vector:
            self.assertEqual(lfu.access(key), expected)

        for key, expected in self.test_vector:
            self.assertEqual(lru.access(key), expected)

    def test_worst_case_lfu(self):
        """
        Worst case for LFU are frequency switches - patterns that are popular and later obsolete. In the case below
        initially followed 3 hit pattern DOES NOT repeat for next 150 accesses.
        """
        lfu = LFU(budget=3)
        test_vector = ['1'] * 3 + ['a', 'b', 'c'] * 50
        for val in test_vector:
            lfu.access(val)
        self.assertLessEqual(lfu.hitrate, 50)

    def test_lru_on_worst_case_lfu(self):
        """
        Here we demonstrate LRU manages well were LFU fails. Since LRU factors in time it notices changes in patterns
        over time.
        """
        lru = LRU(budget=3)
        test_vector = ['1'] * 3 + ['a', 'b', 'c'] * 50
        for val in test_vector:
            lru.access(val)
        self.assertGreaterEqual(lru.hitrate, 95)

    def test_worst_case_lru(self):
        """
        LRU has 0 resistance to simple seesaw pattern where total envelope is greater than size of cache
        """

        lru = LRU(budget=3)
        test_vector = ["a"] * 10 + ['b', 'c', 'd'] + ["a"] * 3 + ['b', 'c', 'd']
        for val in test_vector:
            lru.access(val)
        self.assertLess(lru.hitrate, 60)

    def test_lfu_vs_lru(self):
        """
        LRU has 0 resistance to simple seesaw pattern where total envelope is greater than size of cache
        """
        lfu = LFU(budget=3)
        lru = LRU(budget=3)
        test_vector = (["a"] * 10 + ['b', 'c'] * 3 + ['c', 'd', 'e', 'f']) * 100
        for val in test_vector:
            lfu.access(val)
            lru.access(val)
        self.assertGreater(lfu.hitrate, lru.hitrate)

    def test_lcr(self):
        lcr = LCR(budget=3)
        lfu = LFU(budget=3)
        lru = LRU(budget=3)
        test_vector = (["z"] * 20 + ["a"] * 10 + ['b', 'c'] * 3 + ['c', 'd', 'e', 'f']) * 300
        for val in test_vector:
            lfu.access(val)
            lru.access(val)
            lcr.access(val)
        self.assertGreater(lfu.hitrate, lru.hitrate)
        self.assertGreater(lcr.hitrate, lfu.hitrate)
        print(lcr.hitrate, lfu.hitrate, lru.hitrate)

    def test_lcr_on_worst_case_lru(self):
        """
        LRU has 0 resistance to simple seesaw pattern where total envelope is greater than size of cache
        """

        lru = LRU(budget=3)
        lcr = LCR(budget=3)
        test_vector = ["a"] * 10 + ['b', 'c', 'd'] + ["a"] * 3 + ['b', 'c', 'd']
        for val in test_vector:
            lru.access(val)
            lcr.access(val)
        self.assertGreater(lcr.hitrate, lru.hitrate)

    def test_lcr_on_worst_case_lfu(self):
        """
        Here we demonstrate LRU manages well were LFU fails. Since LRU factors in time it notices changes in patterns
        over time.
        """
        lfu = LFU(budget=3)
        lru = LRU(budget=3)
        lcr = LCR(budget=3)
        test_vector = ['1'] * 3 + ['a', 'b', 'c'] * 50
        for val in test_vector:
            lfu.access(val)
            lru.access(val)
            lcr.access(val)
        self.assertGreater(lru.hitrate, lfu.hitrate)
        self.assertGreater(lcr.hitrate, lfu.hitrate)
        self.assertGreaterEqual(lcr.hitrate, lru.hitrate)


if __name__ == '__main__':
    unittest.main()

