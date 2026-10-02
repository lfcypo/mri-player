# MRI Player

根据传入的音频数据生成 GE Discovery MR750 3.0T MRI 平台的扫描梯度序列 ~~来实现用磁共振听歌喵（bushi~~

> [!CAUTION]
> 磁共振设备需要由专业技师或研究员操作 请勿在不了解设备限制和安全要求的情况下尝试 错误的操作流程和梯度序列会导致设备报废以及人身安全风险

> [!WARNING]
> 因为~~怕被打似喵~~还没拿到操作授权 本项目尚未在 GE Discovery MR750 3.0T 真机平台上进行测试 实际效果不保证 如果你不了解 请勿直接照搬到设备上运行

[GE Discovery MR750 磁共振](https://www.gehealthcare.com/en-au/products/magnetic-resonance-imaging/3-0t/discovery-mr750)

## 使用方法

```sh
uv sync
uv run mri-player convert music.wav \
  --config config/mr750.toml \
  --output-dir output/music
```

`config/mr750.toml` 里的参数请根据实际设备和场地参数调整

## 代码检查

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## 许可

本项目遵守 MIT License

## 致谢

- [qinweizhang/MRI_music_pub](https://github.com/qinweizhang/MRI_music_pub)

## 修正联系

若对项目有任何问题或建议 请提出 issue 或联系学术通讯邮箱 research@lfcypo.com

## 严重警告

本项目输出涉及 MRI 梯度系统及序列控制

未经设备厂商 磁共振物理师 医学工程人员或具备相应资质的科研人员审核与验证 任何生成结果都不应被视为可安全执行的扫描序列

错误或超出硬件限制的梯度波形可能触发设备保护 扫描中止 严重情况下可能造成梯度系统或相关硬件损坏 并带来听力损伤 外周神经刺激等人身安全风险

请仅在获得设备所属单位授权 完成安全审查 并确认所有硬件限制与保护机制均满足要求后进行测试 严禁绕过 关闭或修改设备原有的安全联锁与保护限制
