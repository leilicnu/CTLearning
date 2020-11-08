import numpy as np
import matplotlib.pyplot as plt


class Geometry:
    def __init__(self):
        self.geom = None
    
class ParallelBeamGeometry(Geometry):
    def __init__(self, r=None, phi=None):
        self.geom = "parallel-beam"
        self.r = r
        self.phi = phi
        if self.r==None:
            self.r = np.linspace(-1,1,256)#r 从-1到1的256个值
        if self.phi==None:
            self.phi = np.linspace(0,np.pi,180)#phi 从0到pi的180个方向
    def cbp_filter(self, projection, h):
        """
        平行束的CBP滤波
            projection : 投影数据
            h : 滤波器
        projection_cbp : 滤波后投影
        """
        [m, n] = len(self.phi), len(self.r)
        if h=="RL":                #RL滤波
            h=h_RL(n)
        elif h=="CL":              #CL滤波
            h=h_CL(n)         
        else:                      #不做滤波
            return projection 
        projection_cbp=np.zeros((m,n))  
        #卷积
        for i in range(n):
            for k in range(n):
                if 0<=k-i+n/2<n:
                    projection_cbp[:,i] += projection[:,k]*h[k-i+int(n/2)]
        return projection_cbp
    def rec_cbp(self, projection,N=256):
        """
        平行束的CBP重建
            projection : 投影数据
            N : 待重建图像大小. 默认 256.
        I : 重建后图像
        """
        [m, n] = len(self.phi), len(self.r)
        I = np.zeros((N,N))
        for i in range(m):         #第i个角度的投影
            c = np.cos(np.pi/180*i)
            s = np.sin(np.pi/180*i)
            cm = N/2*(1-c-s)
            for k1 in range(N):
                for k2 in range(N):
                    xm=cm+k2*c+k1*s
                    nn=int(xm)
                    t=xm-nn
                    nn=max(0,nn)   #限定nn范围(0,n-2)
                    nn=min(n-2,nn)
                    p=(1-t)*projection[i,nn]+t*projection[i,nn+1]#线性插值
                    I[N-1-k1,k2] += p
        return I
        
class Shape:
    def forward_projection(self, geometry:Geometry, projection:np=None):
        pass

class Oval(Shape):
    def __init__(self, x0=0, y0=0, a=1, b=1, theta=0, p=1, transmatrix=None):
        """
        初始化椭圆参数:
            (x0, y0) : 中心坐标. 默认 (0,0)
            (a, b) : 椭圆的短轴与长轴. 默认 (1,1)
            theta : 逆时针旋转角度. 默认 0
            p : 线性衰减系数. 默认 1
            transmatrix: 旋转平移矩阵
        """
        self.x0, self.y0, self.a, self.b, self.theta = x0, y0, a, b, theta
        self.p = p
        self.transmatrix = trans_matrix(self.x0, self.y0, self.theta, D=1)
    def point(self, x, y):
        """
        判断点(x,y)是否在椭圆内,若p=0,则在椭圆外
        """
        [x_, y_] = trans(self.transmatrix, x, y)
        p = 0;
        if (x_**2/self.a**2+y_**2/self.b**2)<1:
            p += self.p
        return p
    def draw(self, n = 256, max_x = 1, max_y = 1):
        """
        绘制椭圆图像
            n : 原始图像像素范围 n * n. 默认 256.
            max_x : x范围(-max_x, max_x). 默认 1.
            max_y : x范围(-max_y, max_y). 默认 1.
        I : np数组, 椭圆图像
        """
        I = np.zeros((n,n))            
        x = np.linspace(-max_x,max_x,n)
        y = np.linspace(-max_y,max_y,n)
        for i in range(n):
            for j in range(n):
                I[n-1-j,i] = self.point(x[i], y[j])
        return I
    def string(self, r, phi):
        """
        计算直线(r,phi)与椭圆相交弦长
        """
        c, s = np.cos(phi), np.sin(phi)
        c1, s1 = np.cos(phi-self.theta), np.sin(phi-self.theta)
        a, b, x0, y0 = self.a, self.b, self.x0, self.y0
        a2, b2 = a**2, b**2
        delta=4*b2*a2*(a2*c1**2+b2*s1**2-(r-x0*c-y0*s)**2)
        if delta<0:
            return 0
        else:
            return self.p*np.sqrt(delta)/(a2*c1**2+b2*s1**2)
    def forward_projection(self, geometry: Geometry, projection: np=None):
        """
        geometry: 为扫描几何。
            对平行束扫描: 
                geometry.geom = "parallel-beam"
                geometry.r: 原点到射线的有向距离向量 r\in(-R, R)
                geometry.phi: 角度向量 phi\in[0,pi]
        projeciton: np数组，保存投影数据
        """
        m, n = len(geometry.phi), len(geometry.r)
        projection = np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                projection[m-1-i,j] = self.string(geometry.r[j], geometry.phi[i])
        return projection

def trans_matrix(x0 = 0, y0 = 0, theta = 0, D = 0):
    """
    计算变换矩阵
        x0 : 水平位移. 默认 0.
        y0 : 垂直位移. 默认 0.
        theta : 逆时针旋转角度. 默认 0.
        若D不为0,则计算逆变换矩阵
    T : 三维旋转平移矩阵    
    """
    T = np.eye(3)
    c, s = np.cos(theta), np.sin(theta)
    if D == 0:
        T[0,0], T[0,1], T[0,2] = c, -s, x0
        T[1,0], T[1,1], T[1,2] = s, c, y0
    else:
        T[0,0], T[0,1], T[0,2] = c, s, -x0*c-y0*s
        T[1,0], T[1,1], T[1,2] = -s, c, x0*s-y0*c
    return T

def trans(T, x, y):
    """
    根据变换矩阵T,计算点(x,y)变换后的坐标
    """
    Z = np.ones((3,1))
    Z[0,0], Z[1,0] = x, y
    Z = np.dot(T,Z)
    return [Z[0,0],Z[1,0]]

def h_RL(n=256,d=1):
    """
    R-L滤波
    n : 一个角度下射线的数量.默认256
        d : 探测器间隔.默认1
    h : 滤波器
    """
    h = np.zeros((n,1))
    for i in range(n):
        if i%2==1:
            h[i] = -1/(np.pi*(i-n/2)*d)**2
    h[int(n/2)]=1/(4*d**2)
    return h

def h_CL(n=256,d=1):
    """
    CL滤波
        n : 一个角度下射线的数量.默认256
        d : 探测器间隔.默认1
    h : 滤波器

    """
    h=np.zeros((n,1))
    for i in range(n):
        h[i]=-2/(np.pi**2*d**2*(4*(i-n/2)**2-1))
    return h
        
if __name__=="__main__":    
    a1 = Oval(0, 0, 0.69, 0.92, 0, 1)
    a2 = Oval(0, -0.0184, 0.6624, 0.874, 0, -0.8)
    a3 = Oval(0.22, 0, 0.11, 0.31, (-18/180)*np.pi, -0.2)
    a4 = Oval(-0.22, 0, 0.16, 0.41, (18/180)*np.pi,-0.2)
    a5 = Oval(0, 0.35, 0.21, 0.25, 0, 0.1)
    a6 = Oval(0, 0.1, 0.046, 0.046, 0, 0.1)
    a7 = Oval(0, -0.1, 0.046, 0.046, 0, 0.1)
    a8 = Oval(-0.08, -0.605, 0.046, 0.023, 0, 0.1)
    a9 = Oval(0, -0.606, 0.023, 0.023, 0, 0.1)
    a10 = Oval(-0.06, -0.605, 0.023, 0.046, 0, 0.1)
    a = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]
    
    B=ParallelBeamGeometry()                          # 平行束
   
    projection=a1.forward_projection(B)               # 投影
    for ai in a[1:]:
        projection += ai.forward_projection(B)
    plt.imshow(projection, cmap=plt.cm.gray)
    
    projection_cbp_RL=B.cbp_filter(projection,"RL")   # CBP滤波
    plt.imshow(projection_cbp_RL, cmap=plt.cm.gray) 
    
    I_cbp_RL=B.rec_cbp(projection_cbp_RL)             # 重建
    plt.imshow(I_cbp_RL, cmap=plt.cm.gray)
    
    I=a1.draw()                                       # 原图
    for ai in a[1:]:
        I += ai.draw()
    plt.imshow(I, cmap=plt.cm.gray)
