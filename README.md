> 《實戰人工智慧之深度強化學習》ISBN：9789865021900 http://books.gotop.com.tw/download/ACD017700

### Sarsa & Q-Learning
Sarsa的動作價值函數Q的更新公式為
<div align=center>

<img src="http://latex.codecogs.com/gif.latex?\Q(s_t,a_t)=Q(s_t,a_t)+ \eta\times(R_{t+1}+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t))" />
</div>
Q-Learnig的更新公式為
<div align=center>

<img src="http://latex.codecogs.com/gif.latex?Q(s_t,a_t)=Q(s_t,a_t)+ \eta\times(R_{t+1}+\gamma \underset{a}{\max} (Q(s_{t+1},a)-Q(s_t,a_t))" />
</div>

* Sarsa演算法會在更新時計算下一個動作$a_{t+1}$，然後用來更新函數，但Q-learning則是以狀態$s_{t+1}$的動作價值函數的最大值更新函數。
* Sarsa是使用下個動作$a_{t+1}$更新動作價值函數Q，所以特徵是Q的更新方式取決於計算$a_{t+1}$的策略。這種特徵又稱為**On-Policy**型。
* Q-learning的動作價值函數Q不需要透過動作的策略決定如何更新，所以這種特性又稱為**Off-Policy**型。由於更新公式未完全符合$\epsilon$-greedy法產生的隨機性，所以動作價值函數的收斂也比Sarsa來得更快。[p.57] 

[Jupyter的動畫繪圖參考](http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-notebooks/)

References:
[1] Sutton, Richard S., et al. "Policy gradient methods for reinforcement learning with function approximation." Advances in neural information processing systems. 2000.
[2] Tanaka, Saori C., et al. "Neural mechanisms of gain–loss asymmetry in temporal discounting." Journal of Neuroscience 34.16 (2014): 5595-5602.

#### 安裝強化學習所需的套件
```python
pip install gym
pip install matplotlib
pip install JSAnimation
pip uninstall pyglet -y
pip install pyglet==1.2.4
conda install -c conda-forge ffmpeg
#[p.70]
```
