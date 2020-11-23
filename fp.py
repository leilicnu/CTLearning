import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift, ifft


class Geometry:
    def __init__(self):
        self.geom = None

class ParallelBeamGeometry(Geometry):
    def __init__(self, r=None, phi=None):
        '''r只能偶数,滤波时除以0'''
        self.geom = "parallel-beam"
        self.r = r
        self.phi = phi
        if self.r is None:
            self.r = np.linspace(-1,1,256)        # r 从-1到1的256个值
        if self.phi is None:
            self.phi = np.linspace(0,np.pi,180)   # phi 从0到pi的180个方向
            
    def cbp_filter(self, projection, h):
        """
        平行束的空间域滤波
            projection : 投影数据
            h : 滤波器
        projection_cbp_filter : 滤波后投影
        """
        [m, n] = len(self.phi), len(self.r)
        projection_cbp_filter = np.zeros((m,n))  
        # 卷积
        for i in range(n):
            for k in range(n):
                if 0 <= k-i+n/2 < n:
                    projection_cbp_filter[:,i] += projection[:,k]*h[k-i+int(n/2)]
        return projection_cbp_filter
    
    def backprojection(self, projection, N = None):
        """
        平行束的反投影
            projection : 投影数据
            N : 待重建图像大小. 默认 256.
        I : 重建后图像
        """
        if N is None:
            N = len(self.r)
        [m, n] = len(self.phi), len(self.r)
        I = np.zeros((N,N))
        for i in range(m):              # 第i个角度的投影
            c = np.cos(self.phi[i])
            s = np.sin(self.phi[i])
            for k1 in range(N):
                for k2 in range(N):
                    r=n/N*((k2-N/2)*c + (k1-N/2)*s + N/2)
                    nn = int(r)
                    t = r - nn
                    if 0<=nn<n-1:       # 限定nn范围(0,n-2)
                        p = (1-t)*projection[i, nn] + t*projection[i, nn+1] # 线性插值
                        I[N-1-k1, k2] += p
        return I[:,::-1]
    
    def rec_cbp(self, projection, N=256, flag='CL'):
        """
        平行束的CBP重建
            projection : 投影数据
            N : 待重建图像大小. 默认 256.
            flag: 滤波器类型
        I : 重建后图像
        """
        n = len(self.r)
        if flag=='RL':
            h = h_RL(n)
        elif flag=='CL':
            h = h_CL(n)
        else:
            h = None
        if h is not None:    
            projection = self.cbp_filter(projection, h)          # 滤波
        I = self.backprojection(projection, N)            #反投影 
        return I
    
    def rec_fbp(self, projection, N=256):
        """
        平行束的FBP斜坡滤波重建
            projection : 投影数据
            N : 待重建图像大小. 默认 256.
        I : 重建后图像
        """
        n = len(self.r)
        w = np.linspace(-np.pi, np.pi, n)
        w = abs(w)
        w = fftshift(w)                            # 移动零频点到频谱中间
        projection_f = fft(projection)             # 频域投影
        projection_f *=w                           # 斜坡滤波
        projection_f = np.real(ifft(projection_f)) # 逆傅里叶变换
        I = self.backprojection(projection_f, N)          # 反投影
        return I
    
    def rec_dbp(self, projection, N=256):
        """
        平行束的DBP希尔伯特滤波重建
            projection : 投影数据
            N : 待重建图像大小. 默认 256.
        I : 重建后图像
        """
        n = len(self.r)
        N0 = int(np.floor(N/2))
        projection_d = np.zeros_like(projection)
        for i in range(n-1):                     # 求偏导
            projection_d[:,i]=projection[:,i+1]-projection[:,i]
        I = self.backprojection(projection_d, N)/(2*np.pi)          # 反投影
        f_I = fft(fft(I).T)
        f_I = fftshift(f_I)
        for i in range(N0):                     #希尔伯特变换
            f_I[:,i] *= -1j
            f_I[:,i+N0] *= 1j
        hf_I = fftshift(f_I)
        I = np.real(ifft(ifft(hf_I).T))
        return I
        
class Shape:
    def draw(self, n = 256, max_x = 1, max_y = 1):
        """
        绘制原始图像
            n : 原始图像像素范围 n * n. 默认 256.
            max_x : x范围(-max_x, max_x). 默认 1.
            max_y : y范围(-max_y, max_y). 默认 1.
        I : np数组, 原始图像
        """
        I = np.zeros((n,n))            
        x = np.linspace(-max_x,max_x,n)
        y = np.linspace(-max_y,max_y,n)
        for i in range(n):
            for j in range(n):
                I[n-1-j,i] = self.point(x[i], y[j])
        return I
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
                projection[m-1-i, j] = self.string(geometry.r[j], geometry.phi[i])
        return projection

class Triangle(Shape):
    def __init__(self, x=[-1,1,0], y=[0,0,1], p=1):
        """
        初始化三角形参数:
            x : 三角形三个顶点的横坐标. 默认[-1,1,0]
            y : 三角形三个顶点的纵坐标. 默认[0,0,1]
            p : 线性衰减系数. 默认 1
        """
        self.x, self.y, self.p = x, y, p
       # 三条直线系数ax+by+c=0
        self.a = [self.y[1]-self.y[0], self.y[2]-self.y[1], self.y[2]-self.y[0]]
        self.b = [self.x[0]-self.x[1], self.x[1]-self.x[2], self.x[0]-self.x[2]]
        self.c = []
        for i in range(3):
            self.c +=[-self.x[i]*self.a[i]-self.y[i]*self.b[i]]
        # 直线与另一个点的位置关系
        self.relationship=[self.point_line(0,self.x[2],self.y[2]), 
                self.point_line(1,self.x[0],self.y[0]), self.point_line(2,self.x[1],self.y[1]) ]      
    def point_line(self, i, x, y):
        """
        判断第i条直线与点(x,y)的位置关系
        i取 0,1,2
        """
        A, B, C = self.a[i], self.b[i], self.c[i]
        d = A*x+B*y+C
        if d < 0:
            return -1
        elif d > 0:
            return 1
        else:
            return 0
    def point(self, x, y):
        """
        判断点(x,y)是否在三角形内,若p=0,则在三角形外
        """
        k = []
        for i in range(3):
            k +=[self.point_line(i, x, y)]
        if k==self.relationship:
            return  self.p
        else:
            return 0
    def string(self, r, phi):
        """
        直线(r,phi)与三角形相交弦长
        """
        c, s = np.cos(phi), np.sin(phi)
        point = set()
        for i in range(3):
            delta = self.a[i]*s - c*self.b[i]
            if abs(delta) >= 10**-10 :      # 有交点
                y = (self.a[i]*r + c*self.c[i])/delta
                x = -(self.c[i]*s + self.b[i]*r)/delta
                t = i+1
                if t==3:
                    t = 0
                x_min = min(self.x[i], self.x[t])
                x_max = max(self.x[i], self.x[t])
                y_min = min(self.y[i], self.y[t])
                y_max = max(self.y[i], self.y[t])
                flag_x, flag_y = 0, 0       # 交点是否介于该线段两点间
                if x_min <= x <= x_max or (x_min==x_max and abs(x-x_min)<10**-10):
                    flag_x = 1
                if y_min <= y <= y_max or (y_min==y_max and abs(y-y_min)<10**-10):
                    flag_y = 1
                if flag_x and flag_y:
                    point.add((x,y))
        point = list(point)
        if len(point)>=2:
            p1, p2 = point[0], point[1]
            s=np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            return self.p*s
        else:
            return 0

class Rectangle(Shape):
    def __init__(self, x0=0, y0=0, a=1, b=1, theta=0, p=1, transmatrix=None):
        """
        初始化矩形参数:
            (x0, y0) : 中心坐标. 默认 (0,0)
            (a, b) : 矩形的长和宽. 默认 (1,1)
            theta : 逆时针旋转角度. 默认 0
            p : 线性衰减系数. 默认 1
            transmatrix: 旋转平移矩阵
        """
        self.x0, self.y0, self.a, self.b, self.theta = x0, y0, a, b, theta
        self.p = p
        self.transmatrix = trans_matrix(self.x0, self.y0, self.theta, D=1)
    def point(self, x, y):
        """
        判断点(x,y)是否在矩形内,若p=0,则在矩形外
        """
        [x_, y_] = trans(self.transmatrix, x, y)
        p = 0
        if abs(x_)<self.a/2 and abs(y_)<self.b/2:
            p = self.p
        return p
    def string(self, r, phi):
        """
        直线(r,phi)与矩形相交弦长
        """
        dx, dy =self.a, self.b
        x0, y0 = self.x0, self.y0
        r = r - np.cos(phi)*x0 - np.sin(phi)*y0
        phi = phi - self.theta 
        s, c = np.sin(phi), np.cos(phi)
        point = []                         # 记录直线与矩形交点
        if phi!=0:
            y1 = (r+c*dx/2)/s              # 直线x=-dx/2
            y2 = (r-c*dx/2)/s              # 直线x=dx/2
            if -dy/2 <= y1 < dy/2:
                point += [(-dx/2, y1)]
            if -dy/2 < y2 <= dy/2:
                point += [(dx/2, y2)]
        if phi!=np.pi/2:
            x1 = (r+s*dy/2)/c              # 直线y=-dy/2
            x2 = (r-s*dy/2)/c              # 直线y=dy/2
            if -dx/2 < x1 <= dx/2:
                point += [(x1, -dy/2)]
            if -dx/2 <= x2 < dx/2:
                point += [(x2, dy/2)]  
        if len(point)==2:
            p1, p2 = point[0], point[1]
            s=np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            return self.p*s
        else:
            return 0
                
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
        p = 0
        if (x_**2/self.a**2+y_**2/self.b**2)<1:
            p = self.p
        return p
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

def h_RL(n=256, d=1):
    """
    R-L滤波
    n : 一个角度下射线的数量.默认256
        d : 探测器间隔.默认1
    h : 滤波器
    """
    h = np.zeros((n,1))
    n0 = int(n/2)
    for i in range(n):
        if i%2==1:
            h[i] = -1/(np.pi*(i-n0)*d)**2
    h[n0]=1/(4*d**2)
    return h

def h_CL(n=256, d=1):
    """
    CL滤波
        n : 一个角度下射线的数量. 默认256.
        d : 探测器间隔. 默认1.
    h : 滤波器

    """
    h = np.zeros((n,1))
    n0=n/2
    for i in range(n):
        h[i] = -2/(np.pi**2*d**2*(4*(i-n0)**2-1))
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
    a11 = Rectangle(-0.3, 0, 0.3, 0.3, 0, 0.8)
    a12 = Rectangle(-0.5, -0.5, 0.1, 0.2, (25/180)*np.pi, 1)
    a13 = Rectangle(0.2, -0.33, 0.04, 0.05, (-25/180)*np.pi, .5)
    a14 = Rectangle(0.2, 0.4, 0.4, 0.05, (100/180)*np.pi, .8)
    a15 = Triangle([0.6, 0.73, 0.6], [0.71, 0.6, 0.51], 1)
    a16 = Triangle([-0.6, -0.73, -.5], [-0.4, -0.3, -0.31], 1)
    a17 = Triangle([0.08, 0.083, -0.09], [-0.08, 0.07, 0.091], 0.3)
    a = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17]
    
    B = ParallelBeamGeometry(np.linspace(-1,1,256), np.linspace(0,np.pi,300))     # 平行束

    projection = a1.forward_projection(B)                  # 投影
    for ai in a[1:]:
        projection += ai.forward_projection(B)
    plt.imshow(projection, cmap=plt.cm.gray)
        
    I_cbp_CL = B.rec_cbp(projection, N=256, flag='CL')     # CBP重建
    plt.imshow(I_cbp_CL, cmap=plt.cm.gray)
    
    I_fbp = B.rec_fbp(projection, 256)                     # FBP重建
    plt.imshow(I_fbp, cmap=plt.cm.gray)
    
    I_dbp = B.rec_dbp(projection, 256)                     # DBP重建
    plt.imshow(I_dbp, cmap=plt.cm.gray)
    
    I = a1.draw(512)                                       # 原图
    for ai in a[1:]:
        I += ai.draw(512)
    plt.imshow(I, cmap=plt.cm.gray)

    
