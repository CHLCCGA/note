# SQL 

---

```sql
"D:\WEB\mysql-8.0.31-winx64\bin\mysql.exe" -u root -p
root123

net  start  mysql80
net  stop  mysql80
```

---

### 基础

>
>DB : database
>
>DBMS : database management system (MySQL, Oracle, DB2, Sqlserver)
>
>SQL : structure query language

###### my.ini  配置文件

###### sql启动

```sql
#1
task manager
service
mysql180
#2 
cmd administritor
net  start  mysql80
net  stop  mysql80
```

###### command line client

```sql
exit 
ctrl + c
```

###### cmd

```sql
mysql -h localhost -P3306 -u root -p
root123
mysql -u root -p #主机省略端口
root123
exit
ctrl+c
```

---

### 命令

```sql
#命令结尾 加 ；& \g
#显示数据库
show databases；
#进入数据库
use 数据库名；
#显示
show tables；
show tables from 库名；
#查看目前在那个库
select database()；
#创建table
creat tabel student_info(
id int,
name varchat(20))
#查看结构
desc table名字；
#查看数据
select*from 表名；
#插入
insert into 表明(id,name)value(1,'john');
#修改
update 表名 set name='ccc' wh
```

###### 版本

>```sql
>select version();
>mysql --vers
>mysql -v
>```

###### 语法规范

>不区分大小写 建议关键字大写 表明列明大小写
>
>命令；&、g结尾
>
>命令可换行 缩进
>
>注释 单行#  — 多行/*   */
>
>

###### 图形优化界面

>navicat
>
>sqlyon

---

### DQL(data query language)

#### 基础查询seltct

>select 查询列表 from 表名；
>查询列表 可以是 ：表中字段，常量，表达式，函数
>结果可以是虚拟的表格
>
>```sql
>#查询单个字段
>select last_name from employees;
>
>#多个
>select last_name,salary,email from employees;
>
>#所有  直接sqlyog 里面点要查询的字段
># * 顺序和原始表一模一样 
>select * from employees
>
># `` 1旁边 区分关键字和字段
>
>#查看常量 
>select 100;
>select 'john';
>
>#表达式
>select 100%88;
>
>#查询函数
>select version();
>
># 字段起别名*****************************
>select 100%88 as 结果；
>select last_naem as 性，first_name as 名 from employee;
>
>select last_name 姓， first_naem名 from employee;
>select salary "out put" from employee;
>
>#去重 distinct******************************
>select distinct department_id from employee;
>
># + 作用  只有运算符 其中一方是字符 转换成数值 成功则继续运算 否则我转换成0 一个null结果null
>select 100+90; /*等于*/ select 190;
>select last_name+first_name as 姓名 from employee; # X
>#连接拼接************************************
>select concat('a','b','c') as 结果 
>from employee;
>select concat('last_naem','first_name',select ifnull(commisson_pct,0)) as "out put" 
>from employee;
>select ifnull(commisson_pct,0) as 奖金，
>commission_pct
>```
>

#### 条件查询 where

>```sql
>#条件查询
>select
>字段
>from
>表明
>where
>条件;
>
>#条件表达式 < > =  不等于!= <> <= >=  _________________________________
>select
>*
>from
>employee
>where
>salaty>10000000 ;
>
>select
>name,age
>from
>employee
>where
>salaty>10000000 ;
>
>#逻辑表达式  && || ！ 与或非  and or not__________________________
>select
>name
>from
>employee
>where
>salary >100 and salary<=101;
>
>#模糊查询 like  between and    in   is null______________________
>
>select
>*
>from
>employee
>where
>name like '王'; #名字里有王
># 通配符  %a% 包含任意多个 a
>#转义\  '_\_' 
>#转义自己定$   '_$_' escape '$'; 
>select
>*
>from
>employee
>where
>name like '%a% '; 
>
>select
>*
>from
>employee
>where
>name like '__e_a% '; # 名字 第三个字符e 第五个a 
>
># betweeen and
>select
>*
>from
>employee
>where
>name between 100 and 102;
>
># in
>select
>*
>from
>employee
>where
>name in ('layoff_list','promot_list') ; 
>
># is null is not null
>select
>name
>from
>employee
>where
>salary is nul;
>
>select
>name
>from
>employee
>where
>salary is not nul;
>
>#安全等于 <=>
>select
>name
>from
>employee
>where
>salary <=> nul;
>
>select
>name
>from
>employee
>where
>salary <=> 100000000;
>```
>

#### 排序查询  order by 

>```sql
># 排序查询 order by  升序/降序 asc/desc
>select
>字段
>from
>表明
>where
>条件
>order by 
>排序列表(asc/desc);
>
>
>select
>name
>from
>employee
>where
>salary <=> 100000000
>order by 
>salary desc;
> 
>#长度
>select
>name
>from
>employee
>where
>salary <=> 100000000
>order by 
>salary length(last_name) asc;
>
>#长度
>select
>*
>from
>employee
>where
>salary <=> 100000000
>order by 
>salary asc,age desc ;
>
>```

#### 函数

>单行函数：
>
>concat,length,finull
>
>```sql
>select
>func()
>from
>表明
>where
>条件
>order by 
>排序列表(asc/desc);
>
># length()
># concat()
># upper()
># lower()
># substr substring 截取字符串  
>select
>substr('按劳动法吉安卡罗'，2) 'output'; #截取从2 开始
>select
>substr('按劳动法吉安卡罗'，2,3) 'output';#从2开始 截3个
>
># 可以嵌套 concat( upper(),lower())
>select
>concat( upper(substr(last_name,1,1)),lower(substr(last_name,2))
>
># instr 返回索引 找不到 返回0
>select
>instr('爱好看书发卡行打款哈'，'发卡行打款') as 'output'
>
># trum()
>select
>length(trim('  kljl         '))
>
>select
>length(trim('a' from 'aaaaaaaaaakljlaaaaaaaa'))       
>
># lpad() 左填充 长度不够10 填充 够了截取
># rpad()      
>lpad('jljl',10,'*')
>
># replace()        
>```
>
>```sql
>#数学函数
>
># round 四舍五入
>round(1.567)
>roudn(1.423424,4) # 4 位
>
># cell 向上取整
># floor 向下
>
># truncate 截断
>truncate(1.65555,1)
>
># mod 取模 %
>
>```
>
>```sql
>#日期函数
>
># now 当前日期+时间
>select now();
># curdata 日期没时间
># curtime 时间
>
>#制定的部分 年月日时分秒
>select year(now()) 年；
>select year('1999-1-1') 年；
># monthname 英文名字
>select nonthname(now());
>
>#日期字符转日期
>str_to_date('9-11-1999','%m-%d-%Y')
>#日期转字符
>date_format('1999/9/9','%Y年%m月%d日')
>
>select
>*
>from 
>employee
>where
>str_to_date('9-11-1999','%m-%d-%Y');
>```
>
>```sql
>#其他
>
># version 版本
># database 当前库
># user 当前用户
>```
>
>```sql
># 流程控制
># if
>select if(salary>100000000,'ljlj;','jlj;ljkl')
>
># case   
>/*
>case 
>when then
>when then
>else
>end
>*/
>select salary ,id
>case id
>when 10 then salary*10
>when 100 then salary*100
>else salary
>end
>from employee;
>
>select salary ,id
>case id
>when salary>10 then 'A'
>when salary>100 then 'B'
>else 'C'
>end as 工资级别
>from employee;
>```
>
>分组函数：
>
>做统计  count
>
>```sql
>sum
>avg
>max
>min
>count
>#--------------------------------------------------
>select sum(salary)
>from employee;
># distinct
>select sum(distinct salary)
>from employee;
>
>#统计行数
>select count(*) from employee;
>select count(1) from employee;
>
>```
>
>分组查询 order by   having
>
>```sql
>select
>func()
>from
>表明
>where
>条件
>group by 
>列表
>order by 
>排序列表(asc/desc); 
>
>
>select
>id,avg(salary)
>from
>employee
>group by 
>id
>order by 
>avg(salary)  asc;
>
>select count(*),location_id
>from departments
>where 
>email like '%a%'
>group by
>location_id
>having count(*)>3;
>```

#### 连接查询

>sql92
>
>```sql
>select
>name,boyname 
>from
>boy,girl
>where
>girl.boyfriend_id = boy_id
>
>#等值连接
>select
>name,boyname 
>from
>boy,girl
>where
>girl.boyfriend_id = boy_id
>
>select last_name,department_name,city
>from employee e,department d,location l
>where e.'department_id' = l.'department_id'
>and d.'location_id'=l.'location_id';
>
>#非等值连接
>select
>salary, job_level
>from
>employee e, job_grade j
>where
>salary between j.'lowest_salary' and j.'highest_salary'
>and 
>j.'gread_level' = 'A' ;
>
># 自连接 
>select 
>e.employee_id, e.last_name, m.employee, m.last_name
>from
>employee e, employee m
>where
>e.'manager_id'=m.'employee_id';
>```
>
>sql99
>
>```sql
>/*
>select 
>from 表一 连接类型
>join 表二 on 连接条件
>where
>group by
>having
>order by
>*/
>
>/*
>内连接 inner inner 可以省略
>外连接 outer 左外left outer  右外 right outer 全外 full outer
>交叉连接 cross
>*/
>
>#内连接 inner
>#等值
>select last_name, department_name
>from employee e
>inner join department d 
>on e.'department_id' = d.'department_id';
># 非等值
>select salary, grade_level
>from employee e
>join  job_grade g
>on e.'salary' between g.'lowest_salary' and g.'highesr_salary';
># 自连接
>select e.last_name ,m.last_name
>from employee e
>join employee m
>on e.'manager_id' = m.'employee_id';
>
># 外连接 outer 左外left outer  右外 right outer 全外 full outer
>select g.name, b.*
>from girls g
>left outer join boys b
>on g.'boy_friend_name_id' = g.'id'
>where b.'id' is null;
>
># 交叉连接 cross 笛卡尔乘积
>```

子（内）查询 

>select from where having exists 后面
>
>标量子查询 列(多行)子查询 行子查询 表子查询 
>
>
>```sql
>select first_name from employee where
>department_id in(
>select department_id from department
>    where location_id = 1700
>    )
>
>
>#标量子查询 
>select *
>from employee
>where salary>
>    (select salary
>    from employee
>    where last_name = 'abel'
>    );
>#列(多行)子查询 
># 多行操作符 in/not in   any/some    all
>
>select department_id
>from department
>where location_id in (111,222);
>
>select *
>from employee
>where salary < any(111,222)
>
>select *
>from employee
>where salary > all(111,222)
>#行子查询
>select *
>from employee
>where employee_id = (
>select min(employee_id)
>    from employee
>) and salary =(
>	select max(salary)
>    from employee
>)
>select*
>from employee
>where (employee_id,salary)=(
>	select min(employee_id),max(salary)
>    from employee
>);  
>#表子查询 
>select d.* ,(
>	select count(*)
>	from employee e 
>	where e.department_id = d.'department_id'		
>) 个数
>from departments d;
>
>select ag_dep.*, g.'grade_level'
>from (
>	select avg(salary) ag, department_id
>    from employee
>    group by department_id
>) ag_dep
>inner join job_grade g
>on ag_dep.ag between lowest_sal and highest_sal
>
>select exists(select employee_id from employee where salary=10000)
>
>select department_name
>from department s
>where exists(
>	select*
>    from employee e 
>    where d.'department_id'=e.'department_id'
>)
>
>```
>
>分页查询
>
>```sql
>select
>from
>join 
>on
>where
>group by
>having
>order by
>limit offset,size; # offset 起始索引 从0开始 size 条目个数
>
>select* from employee limit 0,5;
>select* from employee limit 5; # offset 省略
>
>```
>
>union 联合查询 多条查询结果合并
>
>```sql
>select * from employee where email like '%s%'
>union
>select * from employee where department_id>90;
>
>select id from t_ca where sex='male'
>union all #  union 默认去重union all 不去重
>select id from t_eu where sex='male'
>```
>
>

---

### DML(data manipulation language)

#### insert

>```sql
>insert into 表(列，...)  values (值,...),(值,...)
>insert into 表 set 列明=值，列名=值，...
>insert into boys(id,name,gender) values(1,ljk,male)
># nullable 写null 或啥都不写
># 省略列明 默认所有列 顺序一致
>insert into boys set id='kljk',gender='male';
>
>
>```
>
>

#### update

>```sql
>#sql92
>update 表1 b1 ,表2 b2
>set 列=值, 列=值,...
>where 连接条件
>and 筛选条件
>#sql99
>update 表1 b1
>inner/left/right join 表2 b2
>on
>set 列=值, 列=值,...
>where 连接条件
>and 筛选条件
>
>update girl set last_name='lkjl'
>where name like '%sdf% '
>
>update boy b
>inner join girl g on b.'id' = g.'boyfriend_id'
>where b.'boy_name'='jljl'
>
>
>```
>
>
>
>

#### delete

>delete   truncate 
>
>```sql
>delete from 表 while 
>#sql92
>delete 表1 b1
>from 表1 b1, 表2 b2
>where
>and
>#sql99
>delete 表1 b1, 表2 b2
>from 表1 b1
>inner/left/right join 表2 b2 on 
>where
>and
>
>truncate table 表 
>truncate table boy;
>
>delete from girl where phone_number like '%99';
>
>delete g
>from girl g
>inner join boy b on g.'boy_friend_id'=b.'id'
>where b.'boyname'='jlj'
>```

---

### DDL(data define language)

##### 库

>```sql
>#库
># 创建 create database 库名
>create database jljljk;
>create database if not exists jljljk;
>
># 修改 rename database 库名 to 新库名
>rename database jjj to jjjj ;
>#字符集 alter database jjj character set gbk;
>
># 删除库 drop database 库名
>if exists drop database jjj;
>
>```

##### 表

>```sql
>#创建
>create table 表名 (
>列名 列类型1 【（长度）约束】，
>列名 列类型1 【（长度）约束】    
>)
>create table book(
>id int,
>b_name varchar(20),
>price double, 
>author varchar    
>)
>#修改
>alter table book change column publishdate pubDate datetime;
>#列名
>#类型
>#删除
>#表名
>```
>
>

---

### TCL(transaction control language)
