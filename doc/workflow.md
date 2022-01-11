**使用流程**

- 下载数据到某个文件夹
- 克隆```surface-distance```，并重命名文件夹为```surface_distance```，放在```SVLS```目录下

- 在```SVLS```目录下运行```main.py```

```bash
$ CUDA_VISIBLE_DEVICES=1 python main.py --train_option SVLS_V2 --ckpt_dir ckpt/ckpt_brats19_svls_v2_1234 >> log/logfile.log
```

**拓展**

- 指定可使用的显卡 [链接](https://zhuanlan.zhihu.com/p/166161217)

```bash
CUDA_VISIBLE_DEVICES=1,2 python main.py  # 之后在脚本中可以通过os.environ['CUDA_VISIBLE_DEVICES']获取设置的参数
```

- 使用```screen```防止SSH断开连接导致脚本运行终止 [链接](https://segmentfault.com/a/1190000040008299?sort=votes)

```bash
screen -S window_name  # 新建名为window_name的窗口
# 执行脚本...
# 之后可以断开SSH连接，脚本仍会继续执行
# 重新连接SSH
screen -ls  # 列出目前的所有窗口
screen -r window_name  # 切换到名为window_name的窗口
kill window_name  # 注销名为window_name的窗口
```

