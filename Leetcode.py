# Definition for singly-linked list.
from collections import Counter
import sys
import re
import random
import os
import math
import heapq
from heapq import heapify, heappush, heappop
from heapq import heapify, heappush, heappop,
from collections import deque


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:

    dummy = ListNode(0)
    dummy.next = head
    previous = dummy

    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        while head != None and head.next != None:
            if head.val == val:
                previous.next = head.next
                head = head.next
            else:
                previous = head
                head = head.next
        return dummy.next


class Solution(object):
    def removeElements(self, head, val):
        # First loop
        while head and head.val == val:
            head = head.next

        current = head

        # Second loop
        while current:
            while current and current.next and current.next.val == val:
                current.next = current.next.next
            current = current.next
        return head


# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseList(self, head):
        prev_node = None
        current_node = head
        while current_node:
            next_node = current_node.next
            current_node.next = prev_node
            prev_node = current_node
            current_node = next_node
        return prev_node


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head
        while curr != None:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
        return prev


class RecentCounter:

    def __init__(self):

    def ping(self, t: int) -> int:

        # Your RecentCounter object will be instantiated and called as such:
        # obj = RecentCounter()
        # param_1 = obj.ping(t)


class RecentCounter:
    # Queue
    # deque
    # append
    # popleft
    def __init__(self):
        self.q = deque()

    def ping(self, t: int) -> int:
        self.q.append(t)
        while len(self.q) > 0 and t - self.q[0] > 300:
            self.q.popleft
        return len(self.q)


# Your RecentCounter object will be instantiated and called as such:
# obj = RecentCounter()
# param_1 = obj.ping(t)

class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) == 0:
            return True
        stack = []
        for c in s:
            if c == '(' or c == '{' or c == '[':
                stack.append(c)
            else:
                if len(stack) == 0:
                    return False
                else:  # ) } ]
                    temp = stack.pop()
                    if c == ')':
                        if temp != '(':
                            return False
                    elif c == ']':
                        if temp != '[':
                            return False
                    elif c == '}':
                        if temp != '{':
                            return False

        return True if len(stack) == 0 else False


class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        res = []
        stack = []
        for num in nums2:
            stack.append(num)

        for num in nums1:
            temp = []
            isfound = False
            max = -1

            while len(stack) != 0 and not isfound:
                top = stack.pop()
                if top > num:
                    max = top
                elif top == num:
                    isfound = True
                temp.append(top)
            res.append(max)
            while len(temp) != 0:
                stack.append(temp.pop())
        return res


class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        if len(nums) == 0 or nums == None:
            return False
        num = set(nums)
        if len(num) == len(nums):
            return False
        else:
            return True


class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        if len(s) == 0:
            return t
        table = [0]*26
        for i in range(len(t)):
            if i < len(s):
                table[ord(s[i])-ord('a')] -= 1
            table[ord(t[i])-ord('a')] += 1
        for i in range(26):
            if table[i] != 0:
                return chr(i+97)
        return 'a'


class MyHashSet:

    def __init__(self):
        self.hash_list = [False]*10000000

    def add(self, key: int) -> None:
        self.hash_list[key] = True

    def remove(self, key: int) -> None:
        self.hash_list[key] = False

    def contains(self, key: int) -> bool:
        return self.hash_list[key]
        TypeError: 'bool' object is not subscriptable


class MyHashSet:

    def __init__(self):
        self.hash_list = [0]*10000000

    def add(self, key: int) -> None:
        self.hash_list[key] = 1

    def remove(self, key: int) -> None:
        self.hash_list[key] = 0

    def contains(self, key: int) -> bool:
        if self.hash_list[key] > 0:
            return True
        return False

# Your MyHashSet object will be instantiated and called as such:
# obj = MyHashSet()
# obj.add(key)
# obj.remove(key)
# param_3 = obj.contains(key)


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        minheap = []
        heapify(minheap)
        for num in nums:
            heappush(minheap, num)
            if len(minheap) > k:
                heappop(minheap)
        return minheap[0]


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums.sort()
        return nums[len(nums)-k]


class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        counter = Counter(words)
        hashmap = [(-num, word) for word, num in counter.items()]
        heapq.heapify(hashmap)
        return[heapq.heappop(hashmap)[1] for i in range(k)]


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if head == None:
            return False
        fast = head
        slow = head
        while fast != None and fast.next != None:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        else:
            return False


class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        if people == None:
            return 0
        people.sort()
        i = 0
        j = len(people)-1
        res = 0
        while i <= j:
            if people[i]+people[j] <= limit:
                i += 1
            j -= 1
            res += 1
        return res


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if nums == 0 or nums == None:
            return -1
        l = 0
        r = len(nums)-1
        while l <= r:
            m = l + (r-l)//2
            if nums[m] == target:
                return m
            elif nums[m] > target:
                r = m-1
            else:
                l = m+1
        return -1


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        if len(nums) == 0 or nums == None:
            return 0
        l = 0
        r = len(nums)-1
        while l <= r:
            m = l+(r-l)//2
            if nums[m] == target:
                return m
            elif nums[m] > target:
                r = m-1
            else:
                l = m+1
        return r+1


class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        if nums == 0 or nums == None:
            return -1
        l = 0
        r = len(nums)-1
        while l < r:
            m = l+(r-l)//2
            if nums[m] > nums[m+1]:
                r = m
            else:
                l = m+1
        else:
            return l


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if matrix == None or len(matrix) == 0:
            return False
        row = len(matrix)
        col = len(matrix[0])
        l = 0
        r = row*col-1
        while l <= r:
            m = l+(r-l)//2  # m = (r + l) >> 1
            element = matrix[m//col][m % col]
            if element == target:
                return True
            elif element < target:
                l = m+1
            else:
                r = m-1
        return False


class Solution:
    def fib(self, n: int) -> int:
        if n <= 1:
            return n
        return self.fib(n-1)+self.fib(n-2)

    # with cache function


class Solution:
    @cache  # 利用空间记录可能需要的数据
    def fib(self, n: int) -> int:
        if n <= 1:
            return n
        return self.fib(n-1)+self.fib(n-2)


class Solution:
    def fib(self, n: int) -> int:
        if n <= 1:
            return n
        cache = [0]*(n+1)
        cache[1] = 1
        for i in range(2, n+1):
            cache[i] = cache[i-1]+cache[i-2]
        return cache[n]


class Solution:
    def fib(self, n: int) -> int:
        if n <= 1:
            return n
        cur = 0
        pre1 = 0
        pre2 = 1
        for i in range(2, n+1):
            cur = pre1 + pre2
            pre1 = pre2
            pre2 = cur
        return cur


class Solution:
    cache = {0: 0, 1: 1}

    def fib(self, n: int) -> int:
        if n in self.cache:
            return self.cache[n]
        self.cache[n] = self.cache[n-1]+self.cache[n-2]
        return self.cache[n]


#!/bin/python3


#
# Complete the 'fizzBuzz' function below.
#
# The function accepts INTEGER n as parameter.
#

def fizzBuzz(n):
    # Write your code here
    for i in range(1, n+1):
        if i % 3 == 0 and i % 5 == 0:
            print('FizzBuzz')
        elif i % 3 == 0:
            print('Fizz')
        elif i % 5 == 0:
            print('Buzz')
        else:
            print(i)


if __name__ == '__main__':


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [0] + [amount+1]*amount

        for i in range(amount+1):
            for c in coins:
                if c <= i:
                    dp[i] = min(dp[i-c]+1, dp[i])

        if dp[amount] == amount + 1:
            return -1

        return dp[amount]


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        numCoins = len(coins)

        minCoins = [amount + 1] * (amount + 1)
        minCoins[0] = 0

        for i in range(amount + 1):
            for coin in coins:
                if coin <= i:
                    minCoins[i] = min(minCoins[i], minCoins[i-coin] + 1)
        if minCoins[amount] == amount + 1:
            return -1

        return minCoins[amount]


class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        tmp = 0
        result = float('-inf')

        for i, x in enumerate(nums):
            tmp += x
            if i >= k:
                tmp -= nums[i-k]
            if i >= k-1:
                result = max(tmp, result)
        result = result/k

        return result


class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1]*len(nums)

        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j]+1)

        return max(dp)


class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0]*(n+1) for i in range(m+1)]

        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + ord(s1[i-1])
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        result = 0
        for i in s1:
            result += ord(i)
        for i in s2:
            result += ord(i)
        return result - dp[m][n]*2


class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        pos = neg = 0

        for i in range(len(nums)-1):
            if nums[i+1] - nums[i] > 0:
                pos = neg + 1
            elif nums[i+1] - nums[i] < 0:
                neg = pos + 1

        return max(pos, neg) + 1


class Solution:
    def maxSumAfterOperation(self, nums: List[int]) -> int:
        result = 0
        MaxNoSquare = 0
        MaxOneSquare = 0

        for i in nums:
            MaxOneSquare = max((MaxNoSquare + i*i, MaxOneSquare + i))
            MaxNoSquare = max(0, MaxNoSquare + i)
            result = max(result, MaxOneSquare)

        return result


class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        def paint_cost(n, color):
            if (n, color) in self.memo:
                return self.memo[(n, color)]

            total_cost = costs[n][color]
            if n == len(costs) - 1:
                pass
            elif color == 0:
                total_cost += min(paint_cost(n+1, 1), paint_cost(n+1, 2))
            elif color == 1:
                total_cost += min(paint_cost(n+1, 0), paint_cost(n+1, 2))
            elif color == 2:
                total_cost += min(paint_cost(n+1, 0), paint_cost(n+1, 1))
            self.memo[(n, color)] = total_cost
            return total_cost

        if costs == []:
            return 0
        self.memo = {}
        return min(paint_cost(0, 0), paint_cost(0, 1), paint_cost(0, 2))


class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        for n in reversed(range(len(costs)-1)):
            costs[n][0] += min(costs[n+1][1], costs[n+1][2])
            costs[n][1] += min(costs[n+1][0], costs[n+1][2])
            costs[n][2] += min(costs[n+1][0], costs[n+1][1])

        if len(costs) == 0:
            return 0

        return min(costs[0])


class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        for n in range(1, len(costs)):
            costs[n][0] += min(costs[n-1][1], costs[n-1][2])
            costs[n][1] += min(costs[n-1][0], costs[n-1][2])
            costs[n][2] += min(costs[n-1][0], costs[n-1][1])

        if len(costs) == 0:
            return 0

        return min(costs[len(costs)-1])


class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        n = len(costs)
        if n == 0:
            return 0
        c = len(costs[0])

        for house in range(1, n):
            for color in range(c):
                best = math.inf
                for previous_color in range(c):
                    if color == previous_color:
                        continue
                    best = min(best, costs[house-1][previous_color])
                costs[house][color] += best

        return min(costs[-1])


class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        profit, hold = 0, -prices[0]

        for i in range(1, len(prices)):
            profit = max(profit, hold+prices[i]-fee)
            hold = max(hold, profit-prices[i])

        return profit


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        sold, held, reset = float('-inf'), float('-inf'), 0

        for price in prices:
            pre_sold = sold
            sold = price + held
            held = max(held, reset-price)
            reset = max(reset, pre_sold)

        return max(sold, reset)


class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0

        max_so_far = nums[0]
        min_so_far = nums[0]
        result = max_so_far

        for i in range(1, len(nums)):
            curr = nums[i]
            temp_max = max(curr, max_so_far*curr, min_so_far*curr)
            min_so_far = min(curr, max_so_far*curr, min_so_far*curr)
            max_so_far = temp_max
            result = max(max_so_far, result)

        return result


class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        longest_sequence = 0
        for left in range(len(nums)):
            num_zeros = 0
            for right in range(left, len(nums)):
                if num_zeros == 2:
                    break
                if num_zeros == 0:
                    num_zeros += 1
                if num_zeros <= 1:
                    longest_sequence = max(longest_sequence, right-left+1)

        return longest_sequence


class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        left = 0
        right = 0
        zeros = 0
        longest_sequence = 0

        while right < len(nums):
            if nums[right] == 0:
                zeros += 1

            while zeros == 2:
                if nums[left] == 0:
                    zeros -= 1
                left += 1
            longest_sequence = max(longest_sequence, right-left+1)

            right += 1
        return longest_sequence
    


class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        rows = len(matrix)
        cols = len(matrix[0])

        for r in range(rows):
            for c in range(cols):
                if matrix [r][c] == 0:
                    for i in range(cols):
                        matrix[r][i] ='?' if matrix[r][i] else 0
                    for i in range(rows):
                        matrix[i][c] = '?' if matrix[i][c] else 0


        for r in range(rows):
            for c in range(cols):
                if matrix[r][c] == '?':
                    matrix[r][c] = 0



class Solution:
    def getHint(self, secret: str, guess: str) -> str:

        bull = 0
        cow = 0
        secrets_dic = collections.defaultdict(int)

        for s,g in zip(secret,guess):
            if s == g:
                bull += 1
            else:
                secrets_dic[s] +=1
            
        for i,g in enumerate(guess):
            if secret[i] != g and secrets_dic[g]:
                cow += 1
                secrets_dic[g] -= 1
        
        return "".join((str(bull),'A',str(cow),'B'))



class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dict = {}

        for i,n in enumerate(nums):
            if n in dict:
                return dict[n],i
            else:
                dict[target-n] = i


class Solution(object):
    def convert(self, s, numRows):
        if numRows == 1 or numRows >= len(s):
            return s
        
        rows = [[] for row in range(numRows)]
        index = 0
        step = -1
        for char in s:
            rows[index].append(char)
            if index == 0:
                step = 1
            elif index == numRows - 1:
                step = -1
            index += step

        for i in range(numRows):
            rows[i] = ''.join(rows[i])
        return ''.join(rows)


class Solution:
    # https://blog.csdn.net/L141210113/article/details/87925786
    def isMatch(self, s: str, p: str) -> bool:
        memory = [[False] * (len(p) + 1) for i in range(len(s) + 1)]
        memory[len(s)][len(p)] = True

        for i in range(len(s), -1, -1):
            for j in range(len(p) - 1, -1, -1):
                match = i < len(s) and (s[i] == p[j] or p[j] == ".")

                if (j + 1) < len(p) and p[j + 1] == "*":
                    memory[i][j] = memory[i][j + 2]
                    if match:
                        memory[i][j] = memory[i + 1][j] or memory[i][j]
                elif match:
                    memory[i][j] = memory[i + 1][j + 1]

        return memory[0][0]


class Solution:
    # https://blog.csdn.net/L141210113/article/details/87925786
    def isMatch(self, s: str, p: str) -> bool:
        memory = [[False] * (len(p) + 1) for i in range(len(s) + 1)]
        memory[len(s)][len(p)] = True

        for i in range(len(s), -1, -1):
            for j in range(len(p) - 1, -1, -1):
                match = i < len(s) and (s[i] == p[j] or p[j] == ".")

                if (j + 1) < len(p) and p[j + 1] == "*":
                    memory[i][j] = memory[i][j + 2]
                    if match:
                        memory[i][j] = memory[i + 1][j] or memory[i][j]
                elif match:
                    memory[i][j] = memory[i + 1][j + 1]

        return memory[0][0]