# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库


class moea_NSGA2_templet(ea.MoeaAlgorithm):
    """
moea_NSGA2_templet : class - 多目标进化NSGA-II算法类

算法描述:
    采用NSGA-II进行多目标优化，算法详见参考文献[1]。

参考文献:
    [1] Deb K , Pratap A , Agarwal S , et al. A fast and elitist multiobjective
    genetic algorithm: NSGA-II[J]. IEEE Transactions on Evolutionary
    Computation, 2002, 6(2):0-197.

    """

    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 dirName=None,
                 oneStageResult=None,
                 **kwargs):
        # 先调用父类构造方法
        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing,
                         dirName)
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'NSGA2'
        if self.problem.M < 10:
            self.ndSort = ea.ndsortESS  # 采用ENS_SS进行非支配排序
        else:
            self.ndSort = ea.ndsortTNS  # 高维目标采用T_ENS进行非支配排序，速度一般会比ENS_SS要快
        self.selFunc = 'tour'  # 选择方式，采用锦标赛选择
        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=1)  # 生成部分匹配交叉算子对象
            self.mutOper = ea.Mutinv(Pm=1)  # 生成逆转变异算子对象
        elif population.Encoding == 'BG':
            self.recOper = ea.Xovud(XOVR=1)  # 生成均匀交叉算子对象
            self.mutOper = ea.Mutbin(Pm=None)  # 生成二进制变异算子对象，Pm设置为None时，具体数值取变异算子中Pm的默认值
        elif population.Encoding == 'RI':
            self.recOper = ea.Recsbx(XOVR=1, n=20)  # 生成模拟二进制交叉算子对象
            self.mutOper = ea.Mutpolyn(Pm=1 / self.problem.Dim, DisI=20)  # 生成多项式变异算子对象
        else:
            raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')
        self.oneStageResult = oneStageResult

    def reinsertion(self, population, offspring, NUM):

        """
        描述:
            重插入个体产生新一代种群（采用父子合并选择的策略）。
            NUM为所需要保留到下一代的个体数目。
            注：这里对原版NSGA-II进行等价的修改：先按帕累托分级和拥挤距离来计算出种群个体的适应度，
            然后调用dup选择算子(详见help(ea.dup))来根据适应度从大到小的顺序选择出个体保留到下一代。
            这跟原版NSGA-II的选择方法所得的结果是完全一样的。

        """

        # 父子两代合并
        population = population + offspring
        # 选择个体保留到下一代
        [levels, _] = self.ndSort(population.ObjV, NUM, None, population.CV, self.problem.maxormins)  # 对NUM个个体进行非支配分层
        dis = ea.crowdis(population.ObjV, levels)  # 计算拥挤距离
        population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  # 计算适应度
        chooseFlag = ea.selecting('dup', population.FitnV, NUM)  # 调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体
        return population[chooseFlag]

    # two proposed strategies
    def repair(self, X):
        # repair additivity
        delta_X = self.problem.explained_img_prob_score - (np.sum(X, axis=1) + self.problem.black_img_prob_score)
        weight_X = np.full(X.shape, 1 / X.shape[1])
        resign_X = weight_X * delta_X[:, np.newaxis]
        X = X + resign_X
        if X.min() < self.problem.lb[0]:
            self.problem.lb[:] = X.min()
        if X.max() > self.problem.ub[0]:
            self.problem.ub[:] = X.max()

        # repair consistency
        if self.problem.best_faithful_rank is not None:
            repair_prob = self.currentGen / self.MAXGEN
            rand_num = np.random.rand(X.shape[0])
            for i in range(X.shape[0]):
                if rand_num[i] <= repair_prob:
                    X[i, np.argsort(self.problem.best_faithful_rank)] = np.sort(X[i, :])
        return X

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法类的一些动态参数
        # ===========================准备进化============================
        population.initChrom()  # 初始化种群染色体矩阵
        if self.oneStageResult is not None:
            population.Chrom[0, :] = self.oneStageResult
        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群

        # repair strategies
        population.Chrom = self.repair(population.Chrom)

        self.call_aimFunc(population)  # 计算种群的目标函数值
        [levels, _] = self.ndSort(population.ObjV, NIND, None, population.CV, self.problem.maxormins)  # 对NIND个个体进行非支配分层
        population.FitnV = (1 / levels).reshape(-1, 1)  # 直接根据levels来计算初代个体的适应度
        # ===========================开始进化============================
        while not self.terminated(population):
            # 选择个体参与进化
            offspring = population[ea.selecting(self.selFunc, population.FitnV, NIND)]
            # 对选出的个体进行进化操作
            offspring.Chrom = self.recOper.do(offspring.Chrom)  # 重组
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异

            # partial training strategy
            self.problem.partial_training(offspring)

            # repair strategies
            offspring.Chrom = self.repair(offspring.Chrom)


            self.call_aimFunc(offspring)  # 求进化后个体的目标函数值
            population = self.reinsertion(population, offspring, NIND)  # 重插入生成新一代种群
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果
