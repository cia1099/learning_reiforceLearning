> 《實戰人工智慧之深度強化學習》ISBN：9789865021900 http://books.gotop.com.tw/download/ACD017700

### Sarsa & Q-Learning
Sarsa的動作價值函數Q的更新公式為
<div align=center>

<img src="http://latex.codecogs.com/gif.latex?Q(s_t,a_t)=Q(s_t,a_t)+ \eta\times(R_{t+1}+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t))" />
</div>
Q-Learnig的更新公式為
<div align=center>

<img src="http://latex.codecogs.com/gif.latex?Q(s_t,a_t)=Q(s_t,a_t)+ \eta\times(R_{t+1}+\gamma \underset{a}{\max} (Q(s_{t+1},a)-Q(s_t,a_t))" />
</div>

* Sarsa演算法會在更新時計算下一個動作![formula](http://latex.codecogs.com/gif.latex?a_{t+1})，然後用來更新函數，但Q-learning則是以狀態![formula](http://latex.codecogs.com/gif.latex?s_{t+1})的動作價值函數的最大值更新函數。
* Sarsa是使用下個動作![formula](http://latex.codecogs.com/gif.latex?a_{t+1})更新動作價值函數Q，所以特徵是Q的更新方式取決於計算![formula](http://latex.codecogs.com/gif.latex?a_{t+1})的策略。這種特徵又稱為**On-Policy**型。
* Q-learning的動作價值函數Q不需要透過動作的策略決定如何更新，所以這種特性又稱為**Off-Policy**型。由於更新公式未完全符合![formula](https://render.githubusercontent.com/render/math?math=\epsilon-greedy)法產生的隨機性，所以動作價值函數的收斂也比Sarsa來得更快。[p.57] 
* ![formula](https://render.githubusercontent.com/render/math?math=\gamma)是時間折扣率、![formula](https://render.githubusercontent.com/render/math?math=\eta)是學習率。[p.82]
* 為了避免在不知正確的Q table之前，學習可能因收斂至錯誤的解，因此在智能體學習Q table的亂數條件時，還要補以一個 ![](https://render.githubusercontent.com/render/math?math=\epsilon) 機率的隨機移動，在以 ![](https://render.githubusercontent.com/render/math?math=1-\epsilon) 的機率作 ![](http://latex.codecogs.com/gif.latex?\underset{a}{\max}Q) 最大值的移動；這種手法稱為![formula](https://render.githubusercontent.com/render/math?math=\epsilon-greedy)法。[p.51]
* ![](http://latex.codecogs.com/gif.latex?R_{t+1}) 指的是進入狀態 ![](http://latex.codecogs.com/gif.latex?s_{t+1}) 之後的即時報酬。[p.49]

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

### OpenAI:cartPole game
Reference:
[1] [讀這份參考文獻立刻能採用與實踐強化學習_日文fuck](https://deepage.net/machine_learning/2017/08/10/reinforcement-learning.html)

### 深度強化學習 Deep Q-Network(DQN)
表格表示法(Q table)的Q學習是以表格的row編號對應智能體的狀態，column編號則對應智能體的動作，而表格的值則為動作價值 ![](http://latex.codecogs.com/gif.latex?Q(s,a)) 的值。

動作價值 ![](http://latex.codecogs.com/gif.latex?Q(s_t,a_t)) 則是在時間 t 的狀態 ![](http://latex.codecogs.com/gif.latex?s_t) 之下，採取動作 ![](http://latex.codecogs.com/gif.latex?a_t) 所得的折扣報酬總和。

為了解決「狀態變數過多、各變數過於離散化、表格列數會激增的缺點」的問題才使用深度學習。[p.120]

要使用Q-learning的更新公式，以下公式成立的關係就必須成立：
<div align=center>

<img src="http://latex.codecogs.com/gif.latex?Q(s_t,a_t)=R_{t+1}+\gamma\underset{a}{\max}Q(s_{t+1},a)">
</div>

這公式的源頭來自於**貝爾曼方程式**。意思是，新狀態![](http://latex.codecogs.com/gif.latex?s_{t+1})的狀態價值乘上1step的時間折扣率![](http://latex.codecogs.com/gif.latex?\gamma)，再加上即時報酬![](http://latex.codecogs.com/gif.latex?R_{t+1})的最大總和就是目前的狀代價值。貝爾曼公式成立的前提是學習的對象必須是**馬可夫決策過程(Markov Decision Process)**，馬可夫決策過程就是以目前的狀態 ![](https://render.githubusercontent.com/render/math?math=s_t) 與採取動作 ![](https://render.githubusercontent.com/render/math?math=a_t) 確定下一步的狀態 ![](http://latex.codecogs.com/gif.latex?s_{t+1}) 的系統。[p.49]

輸出層元素的輸出值將為 ![](http://latex.codecogs.com/gif.latex?Q(s_t,a_t)) ，而且也會繼續學習，直到輸出值趨近 ![](http://latex.codecogs.com/gif.latex?R_{t+1}+\gamma\underset{a}{\max}Q(s_{t+1},a))。可以用L2或L1誤差來定義損失函數：[p.122]
<div align=center>

<img src="http://latex.codecogs.com/gif.latex?L(s_t,a_t)=[R_{t+1}+\gamma\underset{a}{\max}Q(s_{t+1},a)-Q(s_t,a_t)]^2" />
</div>