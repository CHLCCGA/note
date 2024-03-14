# LeetCode

---

## Array

### 1. 485 Max Consecutive Ones

`dynamic program`

> ```python
> class solution:
> 
>    def find_one(self,nums:list[int])->int:
> 
>        if nums is None or len(nums)==0:
>            return 0
> 
>        consecutive_one = 0 if nums[0] == 0 else 1
>        max_one = consecutive_one
> 
>        for i in range(1,len(nums)):
>            if nums[i]==1:
>                consecutive_one+=1
>            else:
>                consecutive_one=0
>            max_one= max(max_one,consecutive_one)
>        return max_one
> ```
> 
> [youtube 视频](https://www.youtube.com/watch?v=fREz7nG7vA0&list=PLVCBLinWsHYyYvQlZNAAy81s9z_OezZvl&index=7)

### 2. 283 Move Zeros

`two point`

> ```python
> class Solution:
>    def moveZeroes(self, nums: List[int]) -> None:
>        """
>        Do not return anything, modify nums in-place instead.
>        """
>        index = 0
>        for i in range(len(nums)):
>            if nums[i]!=0:
>                nums[index]=nums[i]
>                index+=1
> 
>        for i in range(index,len(nums)):
>            nums[i]=0
> ```
> 
> [youtube](https://www.youtube.com/watch?v=P-HykjnS3Sg&list=PLVCBLinWsHYyYvQlZNAAy81s9z_OezZvl&index=8)

### 3. 27 Remove Element

two point

> ```python
> class Solution:
>    def removeElement(self, nums: List[int], val: int) -> int:
> 
>        if nums is None or len(nums)==0:
>            return 0
>        l = 0
>        r = len(nums)-1
>        while l<r :
>            while (l<r and nums[l]!=val):
>                l+=1
>            while (l<r and nums[r]==val):
>                r-=1
>            nums[l],nums[r]=nums[r],nums[l]
>        if nums[l]==val:
>            return l
>        else:
>            return l+1
> ```
> 
> [youtube](https://www.youtube.com/watch?v=K5c_d7D_Lf8&list=PLVCBLinWsHYyYvQlZNAAy81s9z_OezZvl&index=9)

---

## Linked List

### 4. 203 Remove Linked List Elements

> ```python
> # Definition for singly-linked list.
> # class ListNode:
> #     def __init__(self, val=0, next=None):
> #         self.val = val
> #         self.next = next
> class Solution:
> 
>    def removeElements(self, head, val):
>        dummy = ListNode(-1)
>        dummy.next= head
>        previous = dummy
>        while head!=None:
>            if head.val == val:
>                previous.next=head.next
>                head = head.next 
>            else:
>                previous = head
>                head = head.next
>        return dummy.next
> ```

### 5. 206 Reverse Linked List

> ```python
> # Definition for singly-linked list.
> # class ListNode:
> #     def __init__(self, val=0, next=None):
> #         self.val = val
> #         self.next = next
> class Solution:
> # time limit exceeded
> def Reverse(self, head, val):
>   dummy = ListNode(0)
>   dummy.next = head
> 
>   while head != None:
>       dummy.next = head.next
>       head.next = head.next.next
>       head.next.next = dummy.next
>   return dummy.next
> # accepted
> class Solution:
>    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
>        prev = None
>        curr = head
>        while curr!= None:
>            next = curr.next
>            curr.next = prev
>            prev = curr
>            curr = next
>        return prev 
> ```

---

## Queue

### 6. 933 Number of Recent Calls

> ```python
> from collections import deque
> class RecentCounter:
>    #Queue
>    #deque
>    #append
>    #popleft()
>    def __init__(self):
>        self.q = deque()
> 
> 
>    def ping(self, t: int) -> int:
>        self.q.append(t)
>        while len(self.q)>0 and t-self.q[0]>3000:
>            self.q.popleft()
>        return len(self.q)
> ```

---

## Stack

### 7. 20 Valid Parentheses

> ```python
> class Solution_1:
>    def isValid(self, s: str) -> bool:
>        ack = []
>        lookfor = {')':'(', '}':'{', ']':'['}
> 
>        for p in s:
>            if p in lookfor.values():  # } ] )
>                ack.append(p)
>            elif ack and lookfor[p] == ack[-1]:  #  [  {  (
>                ack.pop()
>            else:
>                return False
> 
>        return ack == []
> 
> class Solution_2:
>    def isValid(self, s: str) -> bool:
>        if len(s) == 0:
>            return True
>        stack = []
>        for c in s:
>            if c == '(' or c == '{' or c == '[':
>                stack.append(c)
>            else:
>                if len(stack) == 0:
>                    return False
>                else:# ) } ]
>                    temp = stack.pop()
>                    if c==')':
>                        if temp!='(':
>                            return False
>                    elif c==']':
>                        if temp!='[':
>                            return False
>                    elif c=='}':
>                        if temp!='{':
>                            return False
> 
>        return True if len(stack)==0 else False
> ```

### 8. 496 Next Greater Element I

> ```python
> class Solution:
>    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
>        res = []
>        stack = []
>        for num in nums2:
>            stack.append(num)
> 
>        for num in nums1:
>            temp = []
>            isfound = False
>            max = -1
> 
>            while len(stack) != 0 and not isfound:
>                top = stack.pop()
>                if top > num:
>                    max = top 
>                elif top == num:
>                    isfound = True
>                temp.append(top)
>            res.append(max)
>            while len(temp)!=0:
>                stack.append(temp.pop())
>        return res
> ```

---

## HashTable

### 9. 217 Contains Duplicate

> ```python
> class Solution:
>    def containsDuplicate(self, nums: List[int]) -> bool:
>        if len(nums)==0 or nums==None:
>            return False
>        num = set(nums)
>        if len(num)==len(nums):
>            return False
>        else:
>            return True
> ```

### 10. 389 Find The Difference

> ```python
> class Solution:
>    def findTheDifference(self, s: str, t: str) -> str:
>            if len(s)==0:
>                return t
>            table = [0]*26
>            for i in range(len(t)):
>                if i <  len(s):
>                    table[ord(s[i])-ord('a')] -= 1
>                table[ord(t[i])-ord('a')] += 1
>            for i in range(26):
>                if table[i]!=0:
>                    return chr(i+97)
> ```

### 11. 705 Design Hashset

> ```python
> class MyHashSet:
> 
>    def __init__(self):
>        self.hash_list =[False]*10000000
> 
> 
>    def add(self, key: int) -> None:
>        self.hash_list[key] =True
> 
> 
>    def remove(self, key: int) -> None:
>        self.hash_list[key] = False
> 
>    def contains(self, key: int) -> bool:
>        return self.hash_list[key]
> ```

----

## Heap

### 12.  215 Kth Largest Element in an Array

> ```python
> from heapq import heapify, heappush, heappop
> class Solution:
>    def findKthLargest(self, nums: List[int], k: int) -> int:
>        minheap = []
>        heapify(minheap)
>        for num in nums:
>            heappush(minheap,num)
>            if len(minheap) > k:
>                heappop(minheap)
>        return minheap[0]
> 
> class Solution:
>    def findKthLargest(self, nums: List[int], k: int) -> int:
>        nums.sort()
>        return nums[len(nums)-k]
> ```

### 13.  692 Top K Frequent Words

> `Hashtable Heap`
> 
> ```python
> from collections import Counter
> from heapq import heapify, heappush, heappop
> import heapq
> class Solution:
>    def topKFrequent(self, words: List[str], k: int) -> List[str]:
>        counter = Counter(words)
>        hashmap = [(-num, word) for word, num in counter.items()] 
>        heapq.heapify(hashmap)
>        return[heapq.heappop(hashmap)[1] for i in range(k) ]
> ```

----

## Two Point

### 14. 141 Linked List Cycle

> ```python
> class Solution:
>    def hasCycle(self, head: Optional[ListNode]) -> bool:
>        if head == None:
>            return False
>        fast = head
>        slow = head
>        while fast != None and fast.next != None:
>            fast = fast.next.next 
>            slow = slow.next
>            if fast == slow:
>                return True
>        else:
>            return False
> ```

### 15. 881 Boats to Save People

> ```python
> class Solution:
>    def numRescueBoats(self, people: List[int], limit: int) -> int:
>        if people == None:
>            return 0
>        people.sort()
>        i = 0
>        j = len(people)-1
>        res = 0
>        while i<=j:
>            if people[i]+people[j]<= limit:
>                i+=1
>            j-=1
>            res +=1
>        return res
> ```

---

## Binary search

### 16. 704 Binary Search

> ```python
> class Solution:
>    def search(self, nums: List[int], target: int) -> int:
>        if nums == 0 or nums == None:
>            return -1
>        l = 0
>        r = len(nums)-1
>        while l <= r:
>            m = l +(r-l)//2  # (l+r)/2 l+r 超出bondary条件 //向下取整
>            if nums[m] == target:
>                return m
>            elif nums[m]>target:
>                r = m-1
>            else:
>                l = m+1
>        return -1
> ```

### 17. 35 Search Insert Position

> ```python
> class Solution:
>    def searchInsert(self, nums: List[int], target: int) -> int:
>        if len(nums)==0 or nums==None:
>            return 0
>        l=0
>        r=len(nums)-1
>        while l<=r:
>            m=l+(r-l)//2
>            if nums[m]==target:
>                return m
>            elif nums[m] >target:
>                r=m-1
>            else:
>                l=m+1
>        return r+1
> ```

### 18. 162 Find Peak Element

> ```python
> class Solution: 
>    def findPeakElement(self, nums: List[int]) -> int:
>        if nums == 0 or nums == None:
>            return -1
>        l = 0 
>        r = len(nums)-1
>        while l<r:
>            m = l+(r-l)//2
>            if nums[m]>nums[m+1]:
>                r = m
>            else:
>                l=m+1
>        else:
>            return l
> ```

### 19. 74 Search a 2D Matrix

> ```python
> class Solution:
>    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
>        if matrix==None or len(matrix)==0:
>            return False
>        row = len(matrix)
>        col = len(matrix[0])
>        l = 0
>        r = row*col-1
>        while l<=r:
>            m = l+(r-l)//2 # m = (r + l) >> 1 位运算
>            element=matrix[m//col][m%col]
>            if element ==target:
>                return True
>            elif element<target:
>                l=m+1
>            else:
>                r=m-1
>        return False
> ```

---

## DP

### 20. 509. Fibonacci number

> ```python
> class Solution:
>    def fib(self, n: int) -> int:
>        if n<=1:
>            return n
>        cur = 0
>        pre1 = 0
>        pre2 = 1
>        for i in range(2,n+1):
>            cur = pre1 +pre2 
>            pre1 = pre2
>            pre2 = cur
>        return cur
> ```
> 
> top-down
> 
> ```python
> class Solution:
>    cache={0:0,1:1}        
>    def fib(self, n: int) -> int:
>        if n in self.cache:
>            return self.cache[n]
>        self.cache[n]= self.cache[n-1]+self.cache[n-2]
>        return self.cache[n]
> ```

### 21. 322 Coin Change

> Button-Up
> 
> ```python
> class Solution:
>    def coinChange(self, coins: List[int], amount: int) -> int:
>            dp = [0] + [amount+1]*amount
> 
>            for i in range(amount+1):
>                for c in coins:
>                    if c <= i:
>                        dp[i] = min(dp[i-c]+1,dp[i])
> 
>            if dp[amount] == amount + 1:
>                return -1
> 
>            return dp[amount]
> ```

### 22. 643. Maximum Average Subarray I

> ```python
> class Solution:
>    def findMaxAverage(self, nums: List[int], k: int) -> float:
>        tmp = 0
>        result = float('-inf')
> 
>        for i,x in enumerate(nums):
>            tmp += x
>            if i>=k:
>                tmp -= nums[i-k]
>            if i>=k-1:
>                result = max(tmp,result)
>        result = result/k
> 
>        return result
> ```

---

## Recurrsion

### 23. 509. Fibonacci number

> ```python
> class Solution:
>    def fib(self, n: int) -> int:
>        if n<=1:
>            return n
>        return self.fib(n-1)+self.fib(n-2)
> ```
> 
> @cache
> 
> ```python
> # with cache function
> class Solution:
>    @cache #利用空间记录可能需要的数据
>    def fib(self, n: int) -> int:
>        if n<=1:
>            return n
>        return self.fib(n-1)+self.fib(n-2)
> ```
> 
> 优化版
> 
> ```python
> class Solution:
>    def fib(self, n: int) -> int:
>        if n<=1:
>            return n
>        cache = [0]*(n+1)
>        cache[1]=1
>        for i in range(2,n+1):
>            cache[i]= cache[i-1]+cache[i-2]
>        return cache[n]
> ```

---

# LeetCode

### 24.fizzBuzz

> ```python
> #!/bin/python3
> 
> import math
> import os
> import random
> import re
> import sys
> 
> 
> #
> # Complete the 'fizzBuzz' function below.
> #
> # The function accepts INTEGER n as parameter.
> #
> 
> def fizzBuzz(n):
> # Write your code here
> for i in range(1,n+1):
>    if i%3==0 and i%5==0:
>        print('FizzBuzz')
>    elif i%3==0:
>        print('Fizz')
>    elif i%5==0:
>        print('Buzz')
>    else:
>        print(i)
> if __name__ == '__main__':
> ```

### 25. 300. Longest Increasing Subsequence (DP)

> ```python
> class Solution:
>    def lengthOfLIS(self, nums: List[int]) -> int:
>        dp = [1]*len(nums)
> 
>        for i in range(1,len(nums)):
>            for j in range(i):
>                if nums[i]>nums[j]:
>                    dp[i] = max(dp[i],dp[j]+1)
> 
>        return max(dp)
> ```

### 26. 712. Minimum ASCII Delete Sum for Two Strings (DP)

> ```python
> class Solution:
>    def minimumDeleteSum(self, s1: str, s2: str) -> int:
>        m,n = len(s1),len(s2)
>        dp = [[0]*(n+1) for i in range(m+1)]
> 
>        for i in range(1,m+1):
>            for j in range(1,n+1):
>                if s1[i-1] == s2[j-1]:
>                    dp[i][j] = dp[i-1][j-1] + ord(s1[i-1])
>                else:
>                    dp[i][j] = max(dp[i-1][j],dp[i][j-1])
> 
>        result = 0
>        for i in s1:
>            result += ord(i)
>        for i in s2:
>            result +=ord(i)
>        return result - dp[m][n]*2
> ```

### 27. 376. Wiggle Subsequence(DP)

> ```python
> class Solution:
>    def wiggleMaxLength(self, nums: List[int]) -> int:
>        pos = neg = 0
> 
>        for i in range(len(nums)-1):
>            if nums[i+1] - nums[i] > 0:
>                pos = neg + 1
>            elif nums[i+1] - nums[i] < 0:
>                neg = pos + 1
> 
>        return max(pos,neg) + 1
> ```

### 28. 1746. Maximum Subarray Sum After One Operation (DP)

> ```python
> class Solution:
>    def maxSumAfterOperation(self, nums: List[int]) -> int:
>        result = 0
>        MaxNoSquare = 0
>        MaxOneSquare = 0
> 
>        for i in nums:
>            MaxOneSquare = max((MaxNoSquare + i*i,MaxOneSquare + i))
>            MaxNoSquare = max(0,MaxNoSquare + i)
>            result = max(result,MaxOneSquare)
> 
>        return result
> ```

### 29. 256. Paint House

> ```python
> # recursive tree
> class Solution:
>    def minCost(self, costs: List[List[int]]) -> int:
>        def paint_cost(n,color):
>            total_cost = costs[n][color]
>            if n == len(costs) - 1:
>                pass
>            elif color == 0:
>                total_cost += min(paint_cost(n+1,1),paint_cost(n+1,2))
>            elif color == 1:
>                total_cost += min(paint_cost(n+1,0),paint_cost(n+1,2))
>            elif color == 2:
>                total_cost += min(paint_cost(n+1,0),paint_cost(n+1,1))  
>            return total_cost
> 
>        if costs == []:
>            return 0
> 
>        return min(paint_cost(0,0),paint_cost(0,1),paint_cost(0,2))              
> ```
> 
> ```python
> # recursive tree with memorization
> class Solution:
>    def minCost(self, costs: List[List[int]]) -> int:
>        def paint_cost(n,color):
>            if (n,color) in self.memo:
>                return self.memo[(n,color)]
> 
>            total_cost = costs[n][color]
>            if n == len(costs) - 1:
>                pass
>            elif color == 0:
>                total_cost += min(paint_cost(n+1,1),paint_cost(n+1,2))
>            elif color == 1:
>                total_cost += min(paint_cost(n+1,0),paint_cost(n+1,2))
>            elif color == 2:
>                total_cost += min(paint_cost(n+1,0),paint_cost(n+1,1))  
>            self.memo[(n,color)] = total_cost
>            return total_cost
> 
>        if costs == []:
>            return 0
>        self.memo = {}
>        return min(paint_cost(0,0),paint_cost(0,1),paint_cost(0,2))              
> ```
> 
> ```python
> # DP approach  top-down
> class Solution:
>    def minCost(self, costs: List[List[int]]) -> int:
>        for n in reversed(range(len(costs)-1)):
>            costs[n][0] += min(costs[n+1][1], costs[n+1][2])
>            costs[n][1] += min(costs[n+1][0], costs[n+1][2])
>            costs[n][2] += min(costs[n+1][0], costs[n+1][1])
> 
>        if len(costs) == 0:
>            return 0
> 
>        return min(costs[0])
> ```
> 
> ```python
> #DP buttom-up
> class Solution:
>    def minCost(self, costs: List[List[int]]) -> int:
>        for n in range(1,len(costs)):
>            costs[n][0] += min(costs[n-1][1], costs[n-1][2])
>            costs[n][1] += min(costs[n-1][0], costs[n-1][2])
>            costs[n][2] += min(costs[n-1][0], costs[n-1][1])
> 
>        if len(costs) == 0:
>            return 0
> 
>        return min(costs[len(costs)-1])
> ```

### 30. 265. Paint House II (Hard)

> ```python
> lass Solution:
>    def minCostII(self, costs: List[List[int]]) -> int:
>        n = len(costs) 
>        if n == 0:
>            return 0
>        c = len(costs[0])
> 
>        for house in range(1,n):
>            for color in range(c):
>                best = math.inf
>                for previous_color in range(c):
>                    if color == previous_color:
>                        continue
>                    best = min(best,costs[house-1][previous_color])
>                costs[house][color] += best
> 
>        return min(costs[-1])
> ```

### 31. 714.Best Time to Buy and Sell Stock with TransactionFee

> ```python
> class Solution:
>    def maxProfit(self, prices: List[int], fee: int) -> int:
>        profit, hold = 0, -prices[0]
> 
>        for i in range(1,len(prices)):
>            profit = max(profit,hold+prices[i]-fee)
>            hold = max(hold,profit-prices[i])
> 
>        return profit
> ```

### 32. 309. Best Time to Buy and Sell Stock with Cooldown

> ```python
> class Solution:
>    def maxProfit(self, prices: List[int]) -> int:
>        sold, held, reset = float('-inf'), float('-inf'),0
> 
>        for price in prices:
>            pre_sold = sold
>            sold = price + held
>            held = max(held, reset-price)
>            reset = max(reset, pre_sold)
> 
>        return max(sold, reset)
> ```

### 33. 152. Maximum Product Subarray

> ```python
> class Solution:
>    def maxProduct(self, nums: List[int]) -> int:
>        if len(nums) == 0:
>            return 0
> 
>        max_so_far = nums[0]
>        min_so_far = nums[0]
>        result = max_so_far
> 
>        for i in range(1,len(nums)):
>            curr = nums[i]
>            temp_max = max(curr, max_so_far*curr, min_so_far*curr)
>            min_so_far = min(curr, max_so_far*curr, min_so_far*curr)
>            max_so_far = temp_max
>            result = max(max_so_far, result)
> 
>        return result 
> ```

### 34. 487. Max Consecutive Ones II

> ```python
> # brute force
> class Solution:
>    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
>        longest_sequence =0
>        for left in range(len(nums)):
>            num_zeros = 0
>            for right in range(left,len(nums)):
>                if num_zeros == 2:
>                    break
>                if num_zeros ==0:
>                    num_zeros += 1
>                if num_zeros <= 1:
>                    longest_sequence = max(longest_sequence, right-left+1)
> 
>        return longest_sequence 
> # DP  sliding window
> 
> class Solution:
>    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
>        left = 0
>        right = 0
>        zeros = 0
>        longest_sequence = 0
> 
>        while right <len(nums):
>            if nums[right] == 0:
>                zeros += 1
> 
>            while zeros == 2:
>                if nums[left] == 0:
>                    zeros -= 1
>                left += 1
>            longest_sequence = max(longest_sequence, right-left+1)
> 
>            right += 1
>        return longest_sequence
> ```

### 35. 73. Set Matrix Zeroes

> ```python
> class Solution:
>    def setZeroes(self, matrix: List[List[int]]) -> None:
>        """
>        Do not return anything, modify matrix in-place instead.
>        """
>        rows = len(matrix)
>        cols = len(matrix[0])
> 
>        for r in range(rows):
>            for c in range(cols):
>                if matrix [r][c] == 0:
>                    for i in range(cols):
>                        matrix[r][i] ='?' if matrix[r][i] else 0
>                    for i in range(rows):
>                        matrix[i][c] = '?' if matrix[i][c] else 0
> 
> 
>        for r in range(rows):
>            for c in range(cols):
>                if matrix[r][c] == '?':
>                    matrix[r][c] = 0
> ```

### 36.299. Bulls and Cows

> ```python
> class Solution:
>    def getHint(self, secret: str, guess: str) -> str:
> 
>        bull = 0
>        cow = 0
>        secrets_dic = collections.defaultdict(int)
> 
>        for s,g in zip(secret,guess):
>            if s == g:
>                bull += 1
>            else:
>                secrets_dic[s] +=1
> 
>        for i,g in enumerate(guess):
>            if secret[i] != g and secrets_dic[g]:
>                cow += 1
>                secrets_dic[g] -= 1
> 
>        return "".join((str(bull),'A',str(cow),'B'))
> ```

### 37.1.Two Sum

> ```python
> class Solution:
>    def twoSum(self, nums: List[int], target: int) -> List[int]:
>        dict = {}
> 
>        for i,n in enumerate(nums):
>            if n in dict:
>                return dict[n],i
>            else:
>                dict[target-n] = i
> ```

### 38.6. Zigzag Conversion

> ```python
> class Solution(object):
>    def convert(self, s, numRows):
>        if numRows == 1 or numRows >= len(s):
>            return s
> 
>        rows = [[] for row in range(numRows)]
>        index = 0
>        step = -1
>        for char in s:
>            rows[index].append(char)
>            if index == 0:
>                step = 1
>            elif index == numRows - 1:
>                step = -1
>            index += step
> 
>        for i in range(numRows):
>            rows[i] = ''.join(rows[i])
>        return ''.join(rows)
> ```

### 39.8. String to Integer

> ```python
> class Solution:
>     def myAtoi(self, str: str) -> int:
>         value, state, pos, sign = 0, 0, 0, 1
> 
>         if len(str) == 0:
>             return 0
> 
>         while pos < len(str):
>             current_char = str[pos]
>             if state == 0:
>                 if current_char == " ":
>                     state = 0
>                 elif current_char == "+" or current_char == "-":
>                     state = 1
>                     sign = 1 if current_char == "+" else -1
>                 elif current_char.isdigit():
>                     state = 2
>                     value = value * 10 + int(current_char)
>                 else:
>                     return 0
>             elif state == 1:
>                 if current_char.isdigit():
>                     state = 2
>                     value = value * 10 + int(current_char)
>                 else:
>                     return 0
>             elif state == 2:
>                 if current_char.isdigit():
>                     state = 2
>                     value = value * 10 + int(current_char)
>                 else:
>                     break
>             else:
>                 return 0
>             pos += 1
> 
>         value = sign * value
>         value = min(value, 2 ** 31 - 1)
>         value = max(-(2 ** 31), value)
> 
>         return value
> ```

### 39.10. Regular Expression Matching

> ```python
> class Solution:
>     # https://blog.csdn.net/L141210113/article/details/87925786
>     def isMatch(self, s: str, p: str) -> bool:
>         memory = [[False] * (len(p) + 1) for i in range(len(s) + 1)]
>         memory[len(s)][len(p)] = True
> 
>         for i in range(len(s), -1, -1):
>             for j in range(len(p) - 1, -1, -1):
>                 match = i < len(s) and (s[i] == p[j] or p[j] == ".")
> 
>                 if (j + 1) < len(p) and p[j + 1] == "*":
>                     memory[i][j] = memory[i][j + 2]
>                     if match:
>                         memory[i][j] = memory[i + 1][j] or memory[i][j]
>                 elif match:
>                     memory[i][j] = memory[i + 1][j + 1]
> 
>         return memory[0][0]
> ```


