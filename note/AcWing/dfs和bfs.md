递归不得不说是个很重要的算法啊，虽然很暴力但很常用`

`熟练使用递归并有一定程度的理解可以为很多高阶算法的基础铺垫基础`

## 总结

- 画出递归树
- 根据分支判断是指数型还是排列型
- 编写代码



**优化**：递归的深度一定程度上决定了耗时长短，有意识地增加宽度缩小深度能很大程度上带来优化

---



### 递归实现指数型枚举

```python
# code
n = int(input())
st = [0] * 15

def dfs(u):
    # 返回条件
    if u == n:
        for i in range(15):
            if st[i]: print("%d " % (i + 1), end = "")
        print("")    
        return
    
    # 根据y总的话，恢复现场很重要，由此保证父节点公平对待每个子节点
    # 不选即为零
    # st[u] = 0
    dfs(u + 1)
    # 选为一
    st[u] = 1
    dfs(u + 1)
    # 恢复现场
    st[u] = 0
    
dfs(0)
```

**复习一个知识点**

位运算左移`>>`和右移`<<`

左移可以理解为`除以`2的n次方

`2 >> 1` 为 2 除以 2，即可把 ` >> 1` 看出除以 2，同理 ` >> 2` 看作除以 4

右移可以理解成`乘以`2的n次方



> 大佬无处不在

```python
n = int(input())
path = []
def dfs(u):
    # 输出所有结点
    print(" ".join(str(i) for i in path))
    # 返回条件为循环结束
    if u == n + 1: return
    # 循环不等长
    for i in range(u, n + 1):
        path.append(i)
        dfs(i + 1)
        # 恢复现场
        path.pop()

dfs(1)
```

- 分析y总和大佬的代码不难发现，y总的解法为`考虑第n个数选或不选`即2的n次方解，大佬的解法为`考虑第n位放哪个数`，也正因如此，大佬的答案是符合字典序排序的，也像极了接下来的排列型枚举

- 无论是理论上还是实际上都不难意识到，大佬的解法，即排序型枚举在深度上远比指数型枚举浅得多，因此速度更快

- 在分析大佬代码时发现，递归解法对于答案选取的知识：当输出在返回条件内时，输出结果均为叶子结点，当输出在返回条件外时，输出结果为所有节点

- 继续观察大佬的代码，可以注意到大佬并不是真的排列型枚举，因此跳出y总的思路再分析大佬的思路：

  :one: 首先注意大佬的返回条件完全不同，当循环中的数选到n时即返回，也就是每个位放哪个数，而不是每个数选不选

  :two: 其次就是这并非排列型枚举的第二个点，循环是不等长的，即每个节点间的子节点数量是不一的
  
  :three: 最后把恢复现场变成了pop

总的来说，大佬用的还是深搜，但是可以理解成换一种思路，快近一倍速度



#### 谁还用递归做指数型呀？

```python
n = int(input())

for i in range(1 << n):
    for k in range(n):
        if (i >> k & 1):
            print(k + 1, end = " ")
    print('')
```

- **1 << n** 放在 `range()` 中表示枚举 0 到 (2^n - 1) - 1，即用二进制枚举所有可能性
- **i >> k & 1** 从 `i` 中分别取出每种可能性

---



### 递归实现排列型枚举

经过上述的操作以及大佬的历练，排列型枚举实在没什么特别的了，直接上代码

```python
n = int(input())

# 这里记录一下构建两个数组的必要性，在排列型枚举中最主要的一个特点就是有序，而非指数型枚举中仅有的选或不选的关系，因此在记录状态外还需要另外记录每个数值是否被使用过
st, used = [], [False] * 9

def dfs(u):
    if u == n:
        print(" ".join(str(i) for i in st))
        return
    
    # 等长循环，除了叶子节点外，每个节点的子节点数量是一样的
    for i in range(n):
        if not used[i]:
            st.append(i + 1)	# 记录状态
            used[i] = True		# 记录该数值已被使用
            dfs(u + 1)
            
            # 恢复现场
            st.pop()
            used[i] = False
            
 dfs(0)
```



### 递归的深度

在递归模板中进行如下操作：

- if 返回时返回0
- 递归时采用dfs() + 1

参看本题`1414. 和为 K 的最少斐波那契数字数目`

```python
class Solution:
    def findMinFibonacciNumbers(self, k: int) -> int:
        if not k: return 0

        f1, f2 = 1, 1
        while k >= f2:
            f1, f2 = f2, f1 + f2
        
        return self.findMinFibonacciNumbers(k - f1) + 1
```



## DFS

### [1219. 黄金矿工](https://leetcode-cn.com/problems/path-with-maximum-gold/)

```python
class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        ans = 0

        def dfs(x, y, gold) -> None:
            nonlocal ans
            gold += grid[x][y]
            ans = max(ans, gold)

            res, grid[x][y] = grid[x][y], 0
            for nx, ny in [[x + 1, y], [x, y + 1], [x - 1, y], [x, y - 1]]:
                if 0 <= nx <= n - 1 and 0 <= ny <= m - 1 and grid[nx][ny]:
                    dfs(nx, ny, gold)
			
            # 恢复现场
            grid[x][y] = res

        # 循环每一个坐标作为起点
        for i in range(n):
            for j in range(m):
                if grid[i][j]: dfs(i, j, 0)

        return ans
```

- 可以注意到并非每个深搜都需要`返回条件`，本题中地图边界、黄金数量为0都是返回条件
- 这里打表操作放进了循环条件内
- `nonlocal` 表示该变量使用的是函数外的同名变量



将程序变得稍微 `dfs` 一点：

```python
class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        ans = 0 
        
        def dfs(x, y):
            # 返回条件
            if x < 0 or y < 0 or x == m or y == n or not grid[x][y]:
                return 0
            record = grid[x][y]
            grid[x][y] = mx = 0
            
            # 递归
            for dx, dy in (0, 1), (1, 0), (0, -1), (-1, 0):
                mx = max(mx, dfs(x + dx, y + dy))
            
            # 恢复现场
            grid[x][y] = record
            
            return record + mx
                
        for i in range(m):
            for j in range(n):
                ans = max(ans, dfs(i, j))
        
        return ans
```



#### **优化**

> 大佬原话

最终答案必然是一个最长的路径，路径中间的点出发是没有必要搜的，而最长路径起点的特征是它周围的矿很少

```python
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        def dfs(x, y):
            if x < 0 or y < 0 or x == m or y == n or not grid[x][y]:
                return 0
            record = grid[x][y]
            grid[x][y] = 0
            
            # 记录每个点作为起点往后的最大采金值
            mx = max(dfs(x + dx, y + dy) for dx, dy in DIRS)
            grid[x][y] = record
            
            # 返回的是该点与该点作为起点出发的采金值
            return record + mx
        
        # 对起点进行优化
        def helper(x, y):
            return sum((nx:=x+dx) < 0 or (ny:=y+dy) < 0 or nx == m or ny == n or not grid[nx][ny] for dx, dy in DIRS) >= 2
        
        return max(dfs(i, j) if grid[i][j] and helper(i, j) else 0 for i in range(m) for j in range(n))
```

- 最大的优化在于函数 `helper()` 实现的功能：只有当角落的坐标才具备作为出发点的价值，角落的坐标：地图边缘、附近有数量为0的坐标
- 用全局变量记录打表的表



### [1020. 飞地的数量](https://leetcode-cn.com/problems/number-of-enclaves/)

- dfs
- bfs

```python
# dfs
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        d = [(0, 1), (1, 0), (-1, 0), (0, -1)]

        # dfs写简洁一点
        def dfs(x, y) -> None:
            if x < 0 or x >= n or y < 0 or y >= m or not grid[x][y]:
                return
            grid[x][y] = 0

            # 把删筛选条件放返回条件内，递归除非剪枝不要做太多工作
            for dx, dy in d:
                dfs(x + dx, y + dy)
        
        # 先遍历边界，把所有非飞地陆地从地图中删去
        for i in range(n):
            dfs(i, 0)
            dfs(i, m - 1)
        for j in range(m):
            dfs(0, j)
            dfs(n - 1, j)
            
        # 计算图中所有剩余的点
        return sum(grid[x][y] for x in range(1, n - 1) for y in range(1, m - 1))
```

:warning: 需要注意的一点：在图论中，对坐标的条件判断放在返回条件里，不要在递归前判断，程序将简洁很多



```python
# bfs
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        # vis标记该坐标是否判断过
        vis = [[False] * n for _ in range(m)]
        
        # q来记录非飞地陆地，并利用deque实现bfs
        q = deque()
        
        # 遍历地图边界，宽搜第一层
        for i, row in enumerate(grid):
            if row[0]:
                vis[i][0] = True
                q.append((i, 0))
            if row[n - 1]:
                vis[i][n - 1] = True
                q.append((i, n - 1))
        for j in range(1, n - 1):
            if grid[0][j]:
                vis[0][j] = True
                q.append((0, j))
            if grid[m - 1][j]:
                vis[m - 1][j] = True
                q.append((m - 1, j))
        
        # 针对宽搜第一层继续深入
        while q:
            r, c = q.popleft()
            for x, y in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= x < m and 0 <= y < n and grid[x][y] and not vis[x][y]:
                    vis[x][y] = True
                    # 将新找到的陆地加入队列进入更深的宽搜
                    q.append((x, y))
        return sum(grid[i][j] and not vis[i][j] for i in range(1, m - 1) for j in range(1, n - 1))
```



### AC1209.带分数

```python
n = int(input())
st, ans = [True] * 9, 0

def check(b):
    cnt = []
    for i in range(9):
        cnt.append(st[i])
    while b:
        x = b % 10
        b //= 10
        if not x or not cnt[x - 1]:
            return False
        cnt[x - 1] = False
    for c in cnt:
        if c:
            return False
    return True

def dfs_c(u, a, c):
    global ans
    if u >= 9:
        return 
    b = n * c - c * a
    if b and c:
        if check(b):
            ans += 1
    for i in range(9):
        if st[i]:
            st[i] = False
            dfs_c(u + 1, a, c * 10 + i + 1)
            st[i] = True

def dfs_a(u, a):
    if a >= n:
        return
    if a:
        dfs_c(u, a, 0)
    for i in range(9):
        if st[i]:
            st[i] = False
            dfs_a(u + 1, a * 10 + i + 1)
            st[i] = True
            
dfs_a(0, 0)
print(ans)
```

在一个dfs中嵌套一个dfs的位置注意一下，在返回条件后递归前



### AC95.费解的开关

```python
import copy

n = int(input())
right = [1, 1, 1, 1, 1]
d = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]

def turn(x, y):
    for dx, dy in d:
        if 0 <= dx + x < 5 and 0 <= dy + y < 5:
            g[dx + x][dy + y] ^= 1

while n:
    ans = float('inf')
    grid = [list(map(int, input())) for _ in range(5)]
    if n != 1:
        input()
    for i in range(1 << 5):
        res = 0
        g = copy.deepcopy(grid)
        for k in range(5):
            if i >> k & 1:
                turn(0, k)
                res += 1
        for r in range(1, 5):
            for c in range(5):
                if not g[r - 1][c]:
                    turn(r, c)
                    res += 1
        if g[4] == right:
            ans = min(ans, res)
    print(ans) if ans <= 6 else print(-1)
    n -= 1
```



### AcWing 1205. 买不到的数目

```python
def dfs(i, n, m):
    if not i:
        return True
    elif i >= n and dfs(i - n, n, m):
        return True
    elif i >= m and dfs(i - m, n, m):
        return True
    return False

for i in range(1, 1000):
    if not dfs(i, n, m):
        res = i
print(res)
```

对于一个i能不能被n和m凑出来，用dfs来做明显是一个指数型枚举
**但y总**这里用的深搜值得一学，不再是对于n和m取或不取，而是减或不减
另外打表根本推不出上述公式…



#### [21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

> 得好好复习数据结构了呀

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# 递归
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # 边界处理，其中某一条列表为空
        if not list1:
            return list2
        elif not list2:
            return list1
        # 对于更小的值而言寻找它的next
        if list1.val <= list2.val:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
        else:
            list2.next = self.mergeTwoLists(list1, list2.next)
            return list2

# 正向
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        pre = ListNode(-1)
        p = pre
        while list1 and list2:
            if list1.val <= list2.val:
                p.next, list1 = list1, list1.next
            else:
                p.next, list2 = list2, list2.next
            p = p.next
        if list1:
            p.next = list1
        if list2:
            p.next = list2
        return pre.next
```

- 本题用正向思维很难解决，因为如果从一开始就将某一边的next指向另一端势必会丢失一段链表的数据，此时方法有二
  - 使用递归，自底向上求解问题，从末尾元素开始处理即可避免数据丢失
    - 使用递归时只需要考虑边界情况、递归条件即可
  - 新开一个指针自上而下记录较小值



## BFS

- 纯纯的模板

```python
q = [(起始位置)]
while q:
    x, y = q.pop(0)
    逻辑语句
    q.append((x, y))
```



### [429. N 叉树的层序遍历](https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal/)

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if not root:
            return []
        # nodes记录本层的所有节点
        nodes, ans = [root], []
        while nodes:
            # cur记录本层节点的值，nxt记录子节点
            cur, nxt = [], []
            for node in nodes:
                cur.append(node.val)
                nxt += [ch for ch in node.children]
            nodes = nxt
            ans.append(cur)
        return ans
```



### AcWing 844. 走迷宫 

```python
n, m = map(int, input().split())
grid = [list(map(int, input().split())) for _ in range(n)]

def bfs():
    queue, path = [(0, 0)], [[-1] * m for _ in range(n)]
    d = [(1, 0), (0, 1), (0, -1), (-1, 0)]
    path[0][0] = 0
    while queue:
        x, y = queue.pop(0)
        for dx, dy in d:
            i, j = dx + x, dy + y
            if 0 <= i < n and 0 <= j < m and path[i][j] == -1 and not grid[i][j]:
                path[i][j] = path[x][y] + 1
                queue.append((i, j))
    return path[-1][-1]

print(bfs())
```



### [542. 01 矩阵](https://leetcode-cn.com/problems/01-matrix/)

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        n, m = len(mat), len(mat[0])
        d = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        queue = []
        for i in range(n):
            for j in range(m):
                if mat[i][j]:
                    mat[i][j] = -1
                else:
                    queue.append([i, j])
        while queue:
            x, y = queue.pop(0)
            for dx, dy in d:
                if 0 <= (nx := x + dx) < n and 0 <= (ny := y + dy) < m and mat[nx][ny] == -1:
                    mat[nx][ny] = mat[x][y] + 1
                    queue.append([nx, ny])
        return mat
```

- 本题求解的是到某个特定位置的最短距离，首先想到bfs，但如果直接对每个位置求最短距离的话，首先耗时巨大其次答案并不正确，因此考虑从每个特定位置作为出发点进行bfs，熟记bfs模板呀！



### [994. 腐烂的橘子](https://leetcode-cn.com/problems/rotting-oranges/)

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        d = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        queue = []
        for i in range(n):
            for j in range(m):
                # 从腐烂的橘子出发作为出发点
                if grid[i][j] == 2:
                    queue.append([i, j])
        while queue:
            x, y = queue.pop(0)
            for dx, dy in d:
                # 当当前橘子是新鲜橘子时才进行进行侵蚀，并加入队列末尾
                if 0 <= (nx := x + dx) < n and 0 <= (ny := y + dy) < m and grid[nx][ny] == 1:
                    grid[nx][ny] = grid[x][y] + 1
                    queue.append([nx, ny])
        ans = 0
        for i in range(n):
            for j in range(m):
                # 如果遇到新鲜橘子则提前返回-1
                if grid[i][j] == 1:
                    return -1
                ans = max(ans, grid[i][j])
        # 存在没有橘子的情况需要对输出答案进行特判
        # 因为腐烂橘子最开始即为2所以需要减去2才是本题所求的最短时间
        return ans - 2 if ans >= 2 else 0
```



###  [417. 太平洋大西洋水流问题](https://leetcode-cn.com/problems/pacific-atlantic-water-flow/)

```python
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        n, m = len(heights), len(heights[0])
        path, ans = [[0] * m for _ in range(n)], set()
        d = [(0, 1), (1, 0), (-1, 0), (0, -1)]

        def dfs(x, y):
            for dx, dy in d:
                if 0 <= (nx := dx + x) < n and 0 <= (ny := dy + y) < m and path[nx][ny] != path[x][y] and heights[nx][ny] >= heights[x][y]:
                    path[nx][ny] |= path[x][y]
                    if path[nx][ny] == 3:
                        ans.add((nx, ny))
                    dfs(nx, ny)

        for i in range(n):
            if path[i][0] != 2:
                path[i][0] |= 2
                if path[i][0] == 3:
                    ans.add((i, 0))
                dfs(i, 0)
            if path[i][m - 1] != 1:
                path[i][m - 1] |= 1
                if path[i][m - 1] == 3:
                    ans.add((i, m - 1))
                dfs(i, m - 1)
        for j in range(m):
            if path[0][j] != 2:
                path[0][j] |= 2
                if path[0][j] == 3:
                    ans.add((0, j))
                dfs(0, j)
            if path[n - 1][j] != 1:
                path[n - 1][j] |= 1
                if path[n - 1][j] == 3:
                    ans.add((n - 1, j))
                dfs(n - 1, j)
        return [list(a) for a in ans]
```

如果考虑正向每个格子能否流向两大洋时间复杂度过高

因此考虑让雨水逆流，从低往高流，分别考虑流进某大洋的雨水可能出自哪里并标记下来

用二进制标记，便于计算也优化空间

> 大佬们的思路

```python
# 代码行更短思路也相对简单，少了很多特判，更为优雅
DIRS = (0, 1), (0, -1), (-1, 0), (1, 0)
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        m, n = len(heights), len(heights[0])

        def bfs(queue):
            explored = set(queue)
            while queue:
                cur = queue.popleft()
                x, y = divmod(cur, n)
                for dx, dy in DIRS:
                    if 0 <= (nx := x + dx) < m and 0 <= (ny := y + dy) < n and heights[nx][ny] >= heights[x][y] and (nxt := nx * n + ny) not in explored:
                        queue.append(nxt)
                        explored.add(nxt)
            return explored
        
        pacific = bfs(deque([i for i in range(n)] + [i * n for i in range(1, m)]))
        atlantic = bfs(deque([(m - 1) * n + i for i in range(n)] + [(i + 1) * n - 1 for i in range(m - 1)]))
        # 利用集合可以与运算的特点快速得出答案
        return list(list(divmod(p, n)) for p in pacific & atlantic)
```

