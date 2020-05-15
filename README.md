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
* 為了避免在不知正確的Q table之前，學習可能因收斂至錯誤的解，因此在智能體學習Q table的亂數條件時，還要補以一個 ![](https://render.githubusercontent.com/render/math?math=\epsilon) 機率的隨機移動，在以 ![](https://render.githubusercontent.com/render/math?math=1-\epsilon) 的機率作 ![](http://latex.codecogs.com/gif.latex?\underset{a}{\max}Q) 最大值的移動；這種手法稱為![formula](https://render.githubusercontent.com/render/math?math=\epsilon-greedy)法。[p.51]其中 ![](https://render.githubusercontent.com/render/math?math=\epsilon) 機率可以隨著epoch的訓練次數來調低機率： ![](http://latex.codecogs.com/gif.latex?\epsilon=\frac{0.5}{1+epoch}) [p.133]
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

當然最佳的損失函數是綜合了L1與L2損失的**Huber**(smooth-L1)函數為最佳，該函數的特性為：[p.124]
<div align=center>

<img src="http://latex.codecogs.com/gif.latex?Loss(e)=
\begin{cases}
[f(x)-\hat{f}(x)]^2& -1\leq{e}\leq{1}\\
|f(x)-\hat{f}(x)|& |e|>1
\end{cases}\\
\text{where}\ e=f(x)-\hat{f}(x)" />
</div>

smooth-L1避免當誤差較大(大於1)時，輸出值或梯度會變得過大，導致學習過程不穩定。

用於計算指令訊號 ![](http://latex.codecogs.com/gif.latex?Q(s_t,a_t)=R_{t+1}+\gamma\underset{a}{\max}Q(s_{t+1},a)) 的detach()可用來取得神經網路的輸出值。在Pytorch使用detach()會讓該變數之前的計算歷程消失，無法在反向傳播演算法的時候計算微分？在學習連結參數時，指令訊號必須先固定，所以執行detach()，避免對指令訊號微分，但神經網路在預測輸出的 ![](http://latex.codecogs.com/gif.latex?Q(s_t,a_t)) 則不執行detach()才能進行微分，讓 ![](http://latex.codecogs.com/gif.latex?Q(s_t,a_t)) 趨近指令訊號，更新神經網路的連結參數。[p.136, code_5.3]


要讓DQN穩定地學習，必須在建置之際重視四項重點[[1]](#ref5_1)。[p.123]

DQN有兩種，分別是2013年版本[[2]](#ref5_2)與2015年Nature版本[[1]](#ref5_3)。利用小批次(batch)學習建置DQN，相當於2013年版本，而2015年版本則是建立Target Q-Network學習Main Q-Network；總共有兩個Q-Network，Q值最大行動a是由Tearget Q-Network定出。[p.147]

#### Reference:

(ref5_1): [1] Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.

[2] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).


### 深度強化學習演算法地圖
Q-learning與DQN都會學習動作價值函數 ![](https://render.githubusercontent.com/render/math?math=Q(s,a))，但是更新動作價值函數Q時，都得用到動作價值函數Q，這也是導致無法穩定學習的原因之一。[p.145]

#### DDQN(Double DQN)
改善DQN學習不穩定的方法；使用兩個Q-Network改良Q值的更新公式：[[1]](#ref_6_1)
<div align=center>

<img src="http://latex.codecogs.com/gif.latex?a_m=\arg\underset{a}{\max}Q_m(s_{t+1},a)\\
" />

<img src="http://latex.codecogs.com/gif.latex?Q_m(s_t,a_t)=Q_m(s_t,a_t)+\eta \times(R_{t+1}+\gamma Q_t(s_{t+1},a_m)-Q_m(s_t,a_t))" />
</div>

這裡的 ![](https://render.githubusercontent.com/render/math?math=Q_m) 是Main Q-Network，![](https://render.githubusercontent.com/render/math?math=Q_t) 是Target Q-Network；每次更新Main Q都是藉由前幾次的暫存Target Q，在每執行幾個步驟後就更新Target Q-Network一次，讓 ![](https://render.githubusercontent.com/render/math?math=Q_t=Q_m) 。[p.147, code6_2]

#### Dueling Network
中心思想是將Q函數分成只由狀態s決定的部份V(s)和由動作決定的部份Advantage，也就是A(s,a)在進行學習，然後在最後的輸出層加總V(s)與A(s,a)，算出Q(s,a)。[[2]](#ref_6_2)

Advantage函數關係為：
<div align=center>

<img src="http://latex.codecogs.com/gif.latex?A(s,a)=Q(s,a)-V(s)" />
</div>
Dueling Network優於DQN的部份在於連接V(s)的神經網路參數不受動作a影響，能在每個step學習，所以能以少於DQN的回合數完成學習，<span style="background-color:yellow">尤其當可選擇的動作增加，這個優勢也就更加明顯</span>。[p.159]
```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        # Dueling Network
        self.fc3_adv = nn.Linear(n_mid, n_out) #Advantage的部份
        self.fc3_v = nn.Linear(n_mid, 1) #價值V的部份

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        adv = self.fc3_adv(h2)
        v = self.fc3_v(h2).expand(-1, adv.size(1))
        output = v + adv -adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        #利用expand展開(broadcast)成[minibatchx動作數量]

        return output
```
減掉平均值的用意在於避免某個動作的偏好，或是說對所有動作的值 ![](https://render.githubusercontent.com/render/math?math=A(s,a_i)) 都歸零在V(s)的數值上：[p.161]
<div align=center>

<img src="http://latex.codecogs.com/gif.latex?Q(s,a_i)=V(s)+A(s,a_i)-\underset{a}{mean}A(s,a)" />
</div>
雖然Dueling Network與DDQN不同之處只有Net類別，但是學習性能卻因此提升。

#### Prioritized Experience Replay
這個手法可在Q-learning無法順利進行時，挑出需優先學習的狀態s的transition。[[3]](#ref6_3)

優先順序的基準就是價值函數貝爾曼方程式的絕對值誤差。嚴格來說，不能稱為TD誤差，但為了方便說明，姑且用TD誤差暫稱之：
<div align=center>

<img src="http://latex.codecogs.com/gif.latex?TD(t)=|R_{t+1}+\gamma \underset{a}{\max}Q(s_{t+1},a)-Q(s_t,a_t)|" />
</div>

將TD誤差明顯的transition優先學習，藉此縮小價值函數的神經網路的輸出誤差。Prioritized Experience Replay的建置方式有很多種，但目前認為以二元樹儲存TD誤差的方式最快速[[11]](#ref6_11)，這次為了方便了解，改以最簡單的list形式。[p.162]
```python
#一個不屌的隨機取樣方法，依list中元素的大小來決定取樣機率，值越大的取樣機會越多
def get_prioritized_indexes(self, batch_size):
    '''以對應TD誤差的機率取得index，
        原文[p.165]版的取樣法不公平，有空再改良'''

    sum_absolute_td_error = np.sum(np.absolute(self.memory)) #self.memory is a list
    sum_absolute_td_error += TD_ERROR_EPSILON*len(self.memory)

    rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
    rnad_list = np.sort(rand_list) #昇幕排序

    indexes = []
    idx = 0
    tmp_abs_td_error = 0
    for rand_num in rand_list:
        while tmp_abs_td_error < rand_num:
            tmp_abs_td_error += abs(self.memory[idx]) + TD_ERROR_EPSILON
            idx += 1 #會缺失取樣idx=0的機會
        
        #因為取樣至少取樣1，長度會超過list的長度，要修正
        if idx >= len(self.memory):
            idx len(self.memory)-1
        indexes.append(idx)

    return indexes
```
範例代碼中Agent的成員函數`memorize_td_error()`用來增加list長度？因為最後都還是呼叫`Agent.update_td_error_memory()`刷新一個新的TD list，所以`memorized_td_error()`形同虛設？[p.171, code6_4]

Prioritized experience replay像是在改善學習效率的方法。可以使訓練Q函數的經驗中batch取TD值較大的狀態做加速的訓練；或是說TD值偏高，代表該動作價值函數 ![](https://render.githubusercontent.com/render/math?math=Q(s_t,a_t)) 之下，學習效率很低，所以在replay時優先取出，多加練習。這種優先學習效率不彰之處的replay演算法就是PER。[p.145]

#### A2C (Advantage Actor-Critic)[[6]](#ref6_6)
這裡的Advantage學習指的是步進大小，例如以二步之後的更新Q函數：
<div align=center>

<img src="http://latex.codecogs.com/gif.latex?Q(s_t,a_t)\to R_{t+1}+\gamma R_{t+2}+\gamma^2\cdot\underset{a}{\max}Q(s_{t+2},a)">
</div>

但不代表以更多的步進更新就會比較好，因為要以多步之後的值更新，所以挑到不適當動作的機率會增加，無法正確學習的機率也就跟著增加；選擇適當步進的值進行Advantage學習才是一般辦法。[p.176]

傳統Q-learning是價值迭代法的手法，但Actor-Critic則同時使用策略迭代法與價值迭代法。Actor只具備動作數量的輸出量，在CartPole課題裡，Actor的輸出量就只有兩個，會針對狀態 ![](https://render.githubusercontent.com/render/math?math=s_t) 的輸入值輸出各動作的優劣程度；當這個輸出值經過softmax函數轉換，就能得到在狀態 ![](https://render.githubusercontent.com/render/math?math=s_t) 下，採用某種動作的機率 ![](https://render.githubusercontent.com/render/math?math=\pi(s_t,a))，與策略迭代法[ch2.3]所要求得的目的相同。Critic得輸出值為狀態的價值 ![](https://render.githubusercontent.com/render/math?math=V^\pi_{s_t}) 。狀態價值就是在狀態 ![](https://render.githubusercontent.com/render/math?math=s_t) 底下所能得到的折扣報酬總和期望值。[p.177]

Actor這邊需要最大化的值就是在狀態 ![](https://render.githubusercontent.com/render/math?math=s_t) 底下採用連結參數 ![](https://render.githubusercontent.com/render/math?math=\theta) 的神經網路，使動作持續執行之後得到的折扣報酬總和 ![](https://render.githubusercontent.com/render/math?math=J(\theta,s_t))：
<div align=center>

<img src="http://latex.codecogs.com/gif.latex?J(\theta,s_t)=E[\log\pi_{\theta}(a|s)\times(\underbrace{Q^\pi(s,a)-V^\pi_s}_{Advantage=A(s,a)})] \tag{6.1}">
</div>

實際建置時，會計算小批次資料的平均E[]。注意動作價值 ![](http://latex.codecogs.com/gif.latex?Q^\pi(s,a)) 不是與動作a有關的變數，而是當常數使用，狀態價值 ![](http://latex.codecogs.com/gif.latex?V^\pi_s)，也是Critic的輸出值。因此我們要求的最大化就是找一個機率使(6.1)式最大，追加策略的熵項：
<div align=center>

<img src="http://latex.codecogs.com/gif.latex?Actor_{entropy}=\sum_a\pi_\theta(a|s)\log\pi_\theta(a|s) \tag{6.2}">
</div>
熵項是隨機挑選動作時(也就是學習初期)的最大值。假設是只能選擇一種動作時，熵項將為最小值？[p.178]

Critic部份會不斷學習，直到能正確輸出狀態價值 ![](http://latex.codecogs.com/gif.latex?V^\pi_s) 為止，所以會盡可能讓採取動作所得的動作價值與輸出一致：
<div align=center>

<img src="http://latex.codecogs.com/gif.latex?loss_{Critic}=[Q^\pi(s,a)-V^\pi_s]^2=A^2(s,a)">
</div>

A2C的建置可以參考OpenAI的A2C範例[[12,13]](#ref6_12)
##### 學習Breakout的四項作業
1. **No-Operation**：這是在重設Breakout的執行環境之後，在0~30步之內什麼都不做，為的是讓遊戲能順利初始化，避免在特定的初始狀態進行學習。
2. **Episodic Life**：Breakout有五個生命數，所以失敗五次，遊戲就結束。允許重複失敗的狀態其實很麻煩，但每次失敗就要全部重來，會無法在多種狀態下學習，因此這個打磚塊遊戲要直接開始下回合的學習，直到失敗五次再徹底重設遊戲。
3. **Max and Skip**：Breakout是以60Hz進行遊戲，會於每4格影格判斷動作(輸入的堆疊為4，所以第一層的Convolution要是4通道)，所以Breakout這個遊戲要在第3格影格與第4格影格預備計算最大值的影像。
4. **Wrap frame**：Breakout的影像是210x160像素、RGB，我們將這影像轉換成DQN的Nature論文使用的1x84x84的灰階影像，影像縮放處理完後還會在用**WrapPytorch**供Pytorch模型的資料型態，讓影像由CxHxW→HxWxC。DQN將報酬介紹在-1~1之間，但Breakout則不需要設定報酬範圍。[p.214-p.219]

#### Reference:

<span id="ref6_1"></span>
[1] Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with double q-learning." Thirtieth AAAI conference on artificial intelligence. 2016.</span>

<span id="ref6_2"></span>
[2] Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." arXiv preprint arXiv:1511.06581 (2015).

<span id="ref6_3"></span>
[3] Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).

<span id="ref6_4"></span>
[4] Mnih, Volodymyr, et al. "Asynchronous methods for deep reinforcement learning." International conference on machine learning. 2016.

[5] ~~Nair, Arun, et al. "Massively parallel methods for deep reinforcement learning." arXiv preprint arXiv:1507.04296 (2015).~~

[6] [OpenAI Baselines:ACKTR & A2C](https://openai.com/blog/baselines-acktr-a2c/)

*--未解說文獻--*
[7] Jaderberg, Max, et al. "Reinforcement learning with unsupervised auxiliary tasks." arXiv preprint arXiv:1611.05397 (2016).

[8] Schulman, John, et al. "Trust region policy optimization." International conference on machine learning. 2015.

[9] Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).

[10] Wu, Yuhuai, et al. "Scalable trust-region method for deep reinforcement learning using kronecker-factored approximation." Advances in neural information processing systems. 2017.

[11] [Let’s make a DQN: Double Learning and Prioritized Experience Replay](https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/)

[12] [A2C](https://github.com/openai/baselines/tree/master/baselines/a2c)

[13] [acktr](https://github.com/openai/baselines/tree/master/baselines/acktr)