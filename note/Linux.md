# Linux小技巧

<br>

## 快速查找文件

- 快速查找文件或文件内容：`ag ""`
  - 这玩意的下载
    1. `brew install wget`
    2. `brew install the_silver_searcher`

<br>

## 下载图片

- 下载图片：`wget --output-document=图片名 "图片地址"`



## 传输文件

- 将source路径下的文件复制到destination中

`scp source destination`

- 一次复制多个文件：

`scp source1 source2 destination`

- 复制文件夹：

`scp -r ~/tmp myserver:/home/acs/`

- 将本地家目录中的tmp文件夹复制到myserver服务器中的/home/acs/目录下。

`scp -r myserver:homework .`

- 将myserver服务器中的~/homework/文件夹复制到本地的当前路径下。

- 指定服务器的端口号：

`scp -P 22 source1 source2 destination`
注意： scp的-r -P等参数尽量加在source和destination之前。



## 管理员

`su-`切换为管理员身份

`sudo -s`命令，回车后输入当前登录的普通用户密码，即可直接获得root权限，进入到root用户模式



## Git

### 全局设置

git config --global user.name xxx：设置全局用户名，信息记录在~/.gitconfig文件中
git config --global user.email xxx@xxx.com：设置全局邮箱地址，信息记录在~/.gitconfig文件中
git init：将当前目录配置成git仓库，信息记录在隐藏的.git文件夹中

### 常用命令

git add XX ：将XX文件添加到暂存区
git commit -m "给自己看的备注信息"：将暂存区的内容提交到当前分支
git status：查看仓库状态
git log：查看当前分支的所有版本
git push -u (第一次需要-u以后不需要) ：将当前分支推送到远程仓库
git clone git@git.acwing.com:xxx/XXX.git：将远程仓库XXX下载到当前目录下
git branch：查看所有分支和当前所处分支

### 查看命令

git diff XX：查看XX文件相对于暂存区修改了哪些内容
git status：查看仓库状态
git log：查看当前分支的所有版本
git log --pretty=oneline：用一行来显示
git reflog：查看HEAD指针的移动历史（包括被回滚的版本）
git branch：查看所有分支和当前所处分支
git pull ：将远程仓库的当前分支与本地仓库的当前分支合并

### 删除命令

git rm --cached XX：将文件从仓库索引目录中删掉，不希望管理这个文件
git restore --staged xx：==将xx从暂存区里移除==
git checkout — XX或git restore XX：==将XX文件尚未加入暂存区的修改全部撤销==

### 代码回滚

git reset --hard HEAD^ 或git reset --hard HEAD~ ：将代码库回滚到上一个版本
git reset --hard HEAD^^：往上回滚两次，以此类推
git reset --hard HEAD~100：往上回滚100个版本
git reset --hard 版本号：回滚到某一特定版本

### 远程仓库

git remote add origin git@git.acwing.com:xxx/XXX.git：将本地仓库关联到远程仓库
git push -u (第一次需要-u以后不需要) ：将当前分支推送到远程仓库
git push origin branch_name：将本地的某个分支推送到远程仓库
git clone git@git.acwing.com:xxx/XXX.git：将远程仓库XXX下载到当前目录下
git push --set-upstream origin branch_name：设置本地的branch_name分支对应远程仓库的branch_name分支
git push -d origin branch_name：删除远程仓库的branch_name分支
git checkout -t origin/branch_name 将远程的branch_name分支拉取到本地
git pull ：将远程仓库的当前分支与本地仓库的当前分支合并
git pull origin branch_name：将远程仓库的branch_name分支与本地仓库的当前分支合并
git branch --set-upstream-to=origin/branch_name1 branch_name2：将远程的branch_name1分支与本地的branch_name2分支对应

### 分支命令

git branch branch_name：创建新分支
git branch：查看所有分支和当前所处分支
git checkout -b branch_name：创建并切换到branch_name这个分支
git checkout branch_name：切换到branch_name这个分支
git merge branch_name：将分支branch_name合并到当前分支上
git branch -d branch_name：删除本地仓库的branch_name分支
git push --set-upstream origin branch_name：设置本地的branch_name分支对应远程仓库的branch_name分支
git push -d origin branch_name：删除远程仓库的branch_name分支
git checkout -t origin/branch_name 将远程的branch_name分支拉取到本地
git pull ：将远程仓库的当前分支与本地仓库的当前分支合并
git pull origin branch_name：将远程仓库的branch_name分支与本地仓库的当前分支合并
git branch --set-upstream-to=origin/branch_name1 branch_name2：将远程的branch_name1分支与本地的branch_name2分支对应

### stash暂存

git stash：将工作区和暂存区中尚未提交的修改存入栈中
git stash apply：将栈顶存储的修改恢复到当前分支，但不删除栈顶元素
git stash drop：删除栈顶存储的修改
git stash pop：将栈顶存储的修改恢复到当前分支，同时删除栈顶元素
git stash list：查看栈中所有元素

<br>

## Else

### 补全指令

`ctrl + r`快速补全历史指令

### 清空命令行指令

- 清空当前界面内的指令

  - `Ctrl L`

  - `clear`

- 清空当前行

  - 使用 Ctrl + U 快捷键：这将清空从光标位置到行首的所有文字。
  - 使用 Ctrl + K 快捷键：这将清空从光标位置到行尾的所有文字。

<br>

## vim

- 命令行模式下的文本编辑器。

- 根据文件扩展名自动判别编程语言。支持代码缩进、代码高亮等功能。

- 使用方式：vim filename
  - 如果已有该文件，则打开它。
  - 如果没有该文件，则打开个一个新的文件，并命名为filename

<br>

### 模式

1.    一般命令模式
      1. 默认模式。命令输入方式：类似于打游戏放技能，按不同字符，即可进行不同操作。可以复制、粘贴、删除文本等。
2.    编辑模式
      1. 在一般命令模式里按下i，会进入编辑模式。
      2. 按下ESC会退出编辑模式，返回到一般命令模式。
3.    命令行模式
      1. 在一般命令模式里按下:/?三个字母中的任意一个，会进入命令行模式。命令行在最下面。
      2. 可以查找、替换、保存、退出、配置编辑器等。

<br>

### 操作

```
(1) i：进入编辑模式

(2) ESC：进入一般命令模式

(3) h 或 左箭头键：光标向左移动一个字符

(4) j 或 向下箭头：光标向下移动一个字符

(5) k 或 向上箭头：光标向上移动一个字符

(6) l 或 向右箭头：光标向右移动一个字符

(7) n<Space>：n表示数字，按下数字后再按空格，光标会向右移动这一行的n个字符

(8) 0 或 功能键[Home]：光标移动到本行开头

(9) $ 或 功能键[End]：光标移动到本行末尾

(10) G：光标移动到最后一行

(11) :n 或 nG：n为数字，光标移动到第n行

(12) gg：光标移动到第一行，相当于1G

(13) n<Enter>：n为数字，光标向下移动n行

(14) /word：向光标之下寻找第一个值为word的字符串。

(15) ?word：向光标之上寻找第一个值为word的字符串。

(16) n：重复前一个查找操作

(17) N：反向重复前一个查找操作

(18) :n1,n2s/word1/word2/g：n1与n2为数字，在第n1行与n2行之间寻找word1这个字符串，并将该字符串替换为word2

(19) :1,$s/word1/word2/g：将全文的word1替换为word2

(20) :1,$s/word1/word2/gc：将全文的word1替换为word2，且在替换前要求用户确认。

(21) v：选中文本

(22) d：删除选中的文本

(23) dd: 删除当前行

(24) y：复制选中的文本

(25) yy: 复制当前行

(26) p: 将复制的数据在光标的下一行/下一个位置粘贴

(27) u：撤销

(28) Ctrl + r：取消撤销

(29) 大于号 >：将选中的文本整体向右缩进一次

(30) 小于号 <：将选中的文本整体向左缩进一次

(31) :w 保存

(32) :w! 强制保存

(33) :q 退出

(34) :q! 强制退出

(35) :wq 保存并退出

(36) :set paste 设置成粘贴模式，取消代码自动缩进

(37) :set nopaste 取消粘贴模式，开启代码自动缩进

(38) :set nu 显示行号

(39) :set nonu 隐藏行号

(40) gg=G：将全文代码格式化

(41) :noh 关闭查找关键词高亮

(42) Ctrl + q：当vim卡死时，可以取消当前正在执行的命令
```

<br>

## 文件信息

`stat` 命令可以显示文件的详细信息，包括创建时间、修改时间、访问时间等。以下是使用 `stat` 命令来查看文件创建时间的示例：

```
stat <文件名>
```

`ls` 命令的 `-lh` 选项。这个选项可以以人类可读的格式显示文件的大小，并将文件大小以 K、M、G 等单位进行转换

```
ls -lh
```

<br>

## 异常处理

​    每次用vim编辑文件时，会自动创建一个.filename.swp的临时文件。

​    如果打开某个文件时，该文件的swp文件已存在，则会报错。此时解决办法有两种：

​        (1) 找到正在打开该文件的程序，并退出

​        (2) 直接删掉该swp文件即可

<br>
