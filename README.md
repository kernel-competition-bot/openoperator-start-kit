# openoperator-start-kit

赛事官网->https://openoperator.cn

快捷监控->http://43.143.241.66:13000

此仓库是openoperator赛事举办方提供的模板仓库。选手可以直接Fork此仓库作为自己队伍的仓库。

## Quick Start

1. Fork此仓库，将仓库可见范围设置为private，将bot帐号加入collaborators
2. 点击`settings`->`webhooks`->`Add webhook`，配置webhook
   1. `Payload URL`填写`http://43.143.241.66:8000/webhook`
   2. `Content type`选择`application/json`
   3. `Secret`填写分配到的密钥（参赛信息收集完成后会分发随机的`webhook secret`）
   4. `SSL verification`选择`Disable`
   5. `Which events...`选择`Just the push event`
   6. 勾选`Active`
   7. 点击`Add webhook`
3. Clone仓库，随便在README.md中写点什么
4. 使用`git push`将修改推送到`github`，恭喜你完成了第一次代码提交！
5. 点开`xx Commits`提交记录页面，边刷新边等待一会，如果系统此时不太忙碌，大约1～3分钟后你就可以在该次提交的评论处看到系统反馈的结果和运行日志。

## 提交说明

### 流程说明

配置好webhook后，当仓库发生push操作，github会向远程服务器发送提交信息。远程仓库检查webhook secret有效性后，拉取仓库更新，执行评估脚本。无论结果如何，执行结束后该次commit评论区会收到日志。如果该次提交的某道题目跑分优于你在该道题目上的历史最好成绩，排行榜的该道题目成绩会更新（排行榜检查是否有新的最好成绩的间隔为5分钟）。

### 仓库结构

提交时仓库根目录需要包含`config`文件和题目的`mlu`代码文件。文件组织结构如下

```bash
.
├── config			# 配置文件，用于指定要评估的题目
├── LeakyReLU.mlu	# bangc代码文件，必须包含kernel函数定义和用于外部程序调用的函数定义
├── ...				# 其他题目的bangc代码文件
└── README.md		# 可选的代码说明
```

> [!NOTE]
>
> 通过`config`文件可以指定本次提交想要评估的题目范围
> config文件的每行代表一个题目，应按照题目序号的三位数字给出
> 例如，LeakyReLU的序号是001，为了评估LeakyReLU题目，config中必须包含一行`001`

> [!TIP]
>
> 每道题目的评估耗时预计不少于30s，评估系统评估完所有题目后才会返回结果，请合理安排评估请求，尽量不要一次性评估太多题目。

> [!CAUTION]
>
> 如果提交中不包含config文件，则会默认评估所有题目！

### 代码要求

1. 代码文件必须以题目名称命名，这是评估脚本能找到你代码的关键要求。
2. 代码中要覆盖头文件引用，核函数定义和用于外部调用的函数定义。
3. 用于外部调用的函数名必须设置为bang_func，bang_func的返回值为`torch::Tensor`，输入参数包含`torch::Tensor input`和参考代码中`__init__`部分定义的其他参数，请参考LeakyReLU示例进行理解。

## 题目&打分

题目按照类别分为`basic`，`easy`，`medium`，`hard`。其中`basic`是必做题，其他类为挑战题。

打分有两个指标：

- 算子结果必须与参考结果误差不大于1e-2，精度达标后性能评估结果才有效
- 性能分数按照`bangc`代码硬件时间相对于`torch`的执行时间赋值

## 最佳实践

1. 每次只评估少量题目
2. 尽量在调试服务器debug，远程评估时通过阅读commit评论中的报错进行debug
3. 系统只接收`main`分支的提交，所以请分时开发或者做好分支管理
4. github评论是执行结束第一时间更新的，快捷监控可以查看目前评估进度，排行榜是周期性更新的，且只会记录团队历史最好成绩
