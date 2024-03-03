# Linux p10

Ubuntu

usernamr@computername

---

### 基本指令

切换输入法：` home+space`

清屏`clear`

download:`sudo apt-get install +<程序名字>chromium-browser`

进入文件:`cd documentsname/foldername` 

返回上级目录:`..`

浏览文件:`ls` 

显示文档信息:`ls -l`

显示全部（包括隐藏文件.hidden）:`ls-a`

显示的比较舒服:	ls - lh

创建文件`touch file.py` 

`touch file1 file2 file3` 创建多个文件

复制文件`cp old new` `cp file1 file1_copy` 

`cp -i file1 file1_copy`重复复制会文件覆盖会提示 

 `-i`interactive  `I` 三个以上提示

`cp file1 folder1/`  `cp file1 file2 folder/` 复制文件到文件夹

`cp -R folder/ folder2/`  文件夹里复制到另一个文件夹  -r -> recursive 

`cp file* folder` 复制 file开头的文件

`cp *4 folder`复制4结尾的文件

移动文件`mv file folder/` 

重命名`mv file file1ename`

创建make direct`mkdir folder3` `mkdir folder3/file3`

移除(空folder)`redir folder3/file3`

移除`rm file3` 

编辑文件 `nano text.py`   `python3 text.py` 用python3运行脚本

显示文件 `cat text.py` `cat text.py > text1.py` 移动文件内容

`cat text.py text.py > text2.py` 两文件内容合并到一个文件

内容放到文件末尾 `cat text3 >> text.py`

执行文件./ `./text.py` 权限要x

---

### 权限管理

`drwxr-xr-x`      r:read w:write e:execute
d:type rwx:user权限  r-x:group权限 r-x:other权限

type  d 文件夹 - 文件

权限  r:read w:write e:execute

修改权限 change mode `chmod u+r text.py` user + read权限
`chmod u-r text.py`user 去点 read权限 
`chmod a-r text.py` a all 全部去掉r权限
`chmod o+r text.py` others 加x权力
`chmod ug+rw text.py` user 和group 加 read write权限

脚本头加上 自动python运行 `#!/user/bin/python3`

---

### MacOs Linux 通过SSH远程

`sudo apt-get install openssh-server` 安装 openssh

mac 输入 `ssh username@加上字符 在linux输入ifconfig inet后面 类似198.168.0.108`

避免每次输入密码  mac `ssh-keygen`
`ssh-copy-id username@192......`

### Windows SSH

`sudo apt-get install openssh-server` 安装 openssh

PuTTY  下载

linux拿到 `ifconfig` inet后面的端口ip copy到putty
如果失败 `sudo apt install nat-tools`

输入 username password

### 手机 Android/IOS

`sudo apt-get install openssh-server` 安装 openssh

手机上找个ssh应用

输入username 和 ip地址

---

### TeamViewer(外网) VNC(局域网) 图像化

teamviewer 直接下就行

VNC  Mac

vnc `sudo apt-get install x11vnc`

` -storepasswd` 
`x11vnc -usepw`
`x11vnc -usepw -forever`

VNC Windows

tightvnc free 

realvnc free
