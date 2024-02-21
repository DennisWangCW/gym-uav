import time
import vtk
import copy
import numpy as np


class Config:
    def __init__(self):
        self.basic_directions = list(np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8]) * np.pi / 16)
        self.extra_directions = list(np.array([-12, -3, -1, 1, 3, 12, 16]) * np.pi / 16)
        self.original_observation_length = 15

        # reward parameters
        self.use_sparse_reward = True
        self.obstacle_coef_sparse = 0.0
        self.action_coef_sparse = 0.001
        self.step_coef_sparse = 0.03
        self.distance_coef_sparse = 0.00
        self.goal_coef_sparse = 50.0
        self.crash_coef_sparse = 50.0

        self.obstacle_coef = 0.2
        self.action_coef = 0.001
        self.step_coef = 0.03
        self.distance_coef = 0.01
        self.goal_coef = 50.0
        self.crash_coef = 50.0

        # curriculum learning parameters
        self.use_curriculum_learning = False
        self.curriculum_height = 400
        self.curriculum_reach_goal_distance = 40
        self.curriculum_min_init_distance = 40
        self.curriculum_map_size = 3

        self.level = 120 # 100.0
        self.max_speed = 50.0
        self.min_distance_to_target = 15 # 10.0
        self.real_action_range = np.array([0.25, 2.0])

        # environment parameters
        self.min_distance_to_obstacle = 1.0
        self.min_initial_starts = 130 # 200.0
        self.expand = 64
        self.num_circle =  8 # 10 ## 15
        self.radius = 60

        # self.period = 200
        self.lowest = 30
        # self.delta = 17
        # self.total = 10

        self.period = 180
        # self.lowest = 40
        self.delta = 23
        self.total = 25

        # range finder parameters
        self.scope = 100.0  # 
        self.min_step = 0.1  # 

        # rendering parameters
        self.margin = 1
        self.camera_alpha = 0.2

        # assert self.min_step < self.min_distance_to_obstacle
        assert self.min_distance_to_obstacle > 0
        assert self.period > self.scope - self.radius


class ToolBox:
    def __init__(self):
        pass

    def SetCamera(self, camera=None, position=(1000,1000,500), direction=0):
        if camera:
            position[0] = position[0] - np.cos(direction / 360 * 2 * np.pi)*40
            position[1] = position[1] - np.sin(direction / 360 * 2 * np.pi)*40
            position[2] = position[2]
            camera.SetPosition(position)

            position[0] = position[0] + np.cos(direction / 360 * 2 * np.pi) * 40
            position[1] = position[1] + np.sin(direction / 360 * 2 * np.pi) * 40
            position[2] = position[2]
            camera.SetFocalPoint(position)

            camera.Elevation(15)
            camera.Azimuth(0)
        else:
            camera = vtk.vtkCamera()
            # camera.SetFocalPoint(0, 0, 100)
            # camera.SetPosition(position)
            # camera.Azimuth(180)
            # camera.Elevation(-90)
            # camera.SetViewAngle(0)
            # camera.ComputeViewPlaneNormal()
            camera.SetViewUp(0, 0, 1)
            camera.Zoom(0.8)
            # position[0] = position[0] - np.cos(direction) * 50
            # position[1] = position[1] - np.sin(direction) * 50
            # position[2] = position[2] + 0
            # camera.SetPosition(position)
            # camera.Azimuth(direction)
            camera.Elevation(200)
            # camera.SetFocalPoint(position)
            return camera

    def CreateGround(self, size=4000):
        # create plane source
        plane = vtk.vtkPlaneSource()
        plane.SetXResolution(100)
        plane.SetYResolution(100)
        plane.SetCenter(0.3, 0.3, 0)
        plane.SetNormal(0, 0, 1)

        # mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())

        # actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        # actor.GetProperty().SetOpacity(0.4)  # 1.0 is totally opaque and 0.0 is completely transparent
        actor.GetProperty().SetColor(211/255,211/255,211/255)
        transform = vtk.vtkTransform()
        transform.Scale(size, size, 1)
        actor.SetUserTransform(transform)

        return actor

    def CreateCoordinates(self, size=1000):
        # create coordinate axes in the render window
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(300, 300, 300)  # Set the total length of the axes in 3 dimensions

        # Set the type of the shaft to a cylinder:0, line:1, or user defined geometry.
        axes.SetShaftType(0)

        axes.SetCylinderRadius(0.02)
        axes.SetSphereRadius(1)
        axes.GetXAxisCaptionActor2D().SetWidth(0.01)
        axes.GetYAxisCaptionActor2D().SetWidth(0.01)
        axes.GetZAxisCaptionActor2D().SetWidth(0.01)

        # axes.SetAxisLabels(0)  # Enable:1/disable:0 drawing the axis labels
        # transform = vtk.vtkTransform()
        # transform.Translate(1.0, 0.0, 0.0)
        # axes.SetUserTransform(transform)
        # axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1,0,0)
        # axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().BoldOff() # disable text bolding
        return axes

    def CreateCylinder(self, p1, p2, r=30, color=(1.0,1.0,1.0), opacity=1.0):
        x, y, h = p1[0], p1[1], p2[2]

        line = vtk.vtkLineSource()
        line.SetPoint1(x, y, 0)
        line.SetPoint2(x, y, h)

        tubefilter = vtk.vtkTubeFilter()
        tubefilter.SetInputConnection(line.GetOutputPort())
        tubefilter.SetRadius(r)
        tubefilter.SetNumberOfSides(30)
        # tubefilter.SetResolution(30)
        tubefilter.CappingOff()
        # tubefilter.CappingOn()

        cylinderMapper = vtk.vtkPolyDataMapper()
        cylinderMapper.SetInputConnection(tubefilter.GetOutputPort())
        cylinderActor = vtk.vtkActor()
        cylinderActor.GetProperty().SetColor(color)
        cylinderActor.GetProperty().SetOpacity(opacity)
        cylinderActor.SetMapper(cylinderMapper)
        return cylinderActor

    def CreateLine(self, p1, p2, color=(56/255,94/255,16/255)):
        line = vtk.vtkLineSource()
        line.SetPoint1(p1)
        line.SetPoint2(p2)

        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputConnection(line.GetOutputPort())
        lineActor = vtk.vtkActor()
        lineActor.GetProperty().SetColor(color)
        lineActor.GetProperty().SetLineWidth(2.0)
        lineActor.SetMapper(lineMapper)

        return lineActor

    def CreateArrow(self, angle=90, scale=(5, 5, 5), position=(100, 100, 100),
                    color=(255 / 255, 0 / 255, 0 / 255)):
        pointer = vtk.vtkArrowSource()
        pointer.SetTipLength(0.15)
        pointer.SetTipRadius(0.08)
        pointer.SetTipResolution(100)
        pointer.SetShaftRadius(0.015)
        pointer.SetShaftResolution(100)
        # pointer.SetCenter()

        transform = vtk.vtkTransform()
        # transform.Scale((0.5, 0.5, 1))
        transform.RotateWXYZ(angle, 0, 0, 1)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputConnection(pointer.GetOutputPort())
        transformFilter.Update()

        pointerActor = vtk.vtkActor()
        pointerActor.SetScale(scale)
        pointerActor.AddPosition(position)
        pointerActor.GetProperty().SetColor(color)

        pointerMapper = vtk.vtkPolyDataMapper()
        pointerMapper.SetInputConnection(transformFilter.GetOutputPort())

        pointerActor.SetMapper(pointerMapper)

        return pointerActor

    def CreateSphere(self, p, r, color=(199/255,97/255,20/255), opacity=1.0):
        ball = vtk.vtkSphereSource()
        ball.SetRadius(r)
        ball.SetCenter(p[0], p[1], p[2])
        ball.SetPhiResolution(16)
        ball.SetThetaResolution(32)

        ballMapper = vtk.vtkPolyDataMapper()
        ballMapper.SetInputConnection(ball.GetOutputPort())
        ballActor = vtk.vtkActor()
        ballActor.GetProperty().SetColor(color)
        ballActor.SetMapper(ballMapper)
        ballActor.GetProperty().SetOpacity(opacity)

        return ballActor

    # def CreateWing(self, color=(199/255,97/255,20/255)):
    #     circle = vtk.vtkRegularPolygonSource()
    #     circle.GeneratePolygonOff()
    #     circle.SetNumberOfSides(5)
    #     circle.SetRadius(500)
    #     circle.SetCenter(0, 0, 0)
    #     # circle.update()
    #     circleMapper = vtk.vtkPolyDataMapper()
    #     circleMapper.SetInputConnection(circle.GetOutputPort())
    #     circleActor = vtk.vtkActor()
    #     circleActor.GetProperty().SetColor(color)
    #     circleActor.SetMapper(circleMapper)
    #
    #     return circleActor


class TimerCallback():
    def __init__(self, renderer):
        self.renderer = renderer
        self.creator = ToolBox()
        self.renderer.SetActiveCamera(self.creator.SetCamera())
        self.terminate_render = False

        self.timerId = None
        self.env_params = None
        self.agent_params = None

        self.env_params_old = None
        self.agent_params_old = None

        self.p1 = list(np.random.randint(0, 500, 2)) + [0]
        self.p2 = list(np.random.randint(0, 500, 2)) + [100]

    def execute(self, iren, event):
        time.sleep(0.001)
        if self.terminate_render:
            iren.DestroyTimer(self.timerId)

        if self.env_params is not None and self.env_params_old is None:
            self.env_params_old = self.env_params
            cylinders = self.env_params['cylinders']
            size = self.env_params['size']

            groundActor = self.creator.CreateGround(size)
            self.renderer.AddActor(groundActor)
            axisActor = self.creator.CreateCoordinates(size)
            self.renderer.AddActor(axisActor)
            for cyl in cylinders:
                cylinderActor = self.creator.CreateCylinder(cyl[0], cyl[1], cyl[2], opacity=0.5)
                self.renderer.AddActor(cylinderActor)

            agent_origin = self.env_params['departure']
            target_origin = self.env_params['arrival']
            try:
                color_departure = self.env_params['color_departure']
                color_destination = self.env_params['color_destination']
            except:
                color_departure = (1.0, 0.0, 0.0)
                color_destination = (0.0, 0.0, 1.0)

            shape = np.shape(agent_origin)
            if len(shape) > 1:
                for i in range(shape[0]):
                    ballActor = self.creator.CreateSphere(agent_origin[i], 5, color=color_departure, opacity=1.0)
                    self.renderer.AddActor(ballActor)
            else:
                ballActor = self.creator.CreateSphere(agent_origin, 5, color=color_departure, opacity=1.0)
                self.renderer.AddActor(ballActor)

            ballActor = self.creator.CreateSphere(target_origin, 5, color=color_destination, opacity=1.0)
            self.renderer.AddActor(ballActor)

            iren.GetRenderWindow().Render()

        if self.agent_params is not None:
            if self.agent_params_old is None:
                self.agent_params_old = self.agent_params

            elif not list(self.agent_params['position']) == list(self.agent_params_old['position']):
                self.agent_params_old = self.agent_params
                agent_origin = self.agent_params['position']
                try:
                    position_camera = self.agent_params['position_camera']
                except:
                    position_camera = self.agent_params['position']
                direction = self.agent_params['direction']
                try:
                    color = self.agent_params['color']
                except:
                    color = (199/255, 97/255, 20/255)
                try:
                    direction_camera = self.agent_params['direction_camera']
                except:
                    direction_camera = self.agent_params['direction']

                # camera_position = copy.deepcopy(agent_origin)
                camera = self.renderer.GetActiveCamera()
                self.creator.SetCamera(camera=camera, position=position_camera, direction=direction_camera)

                ballActor = self.creator.CreateSphere(agent_origin, 3, color=color)
                self.renderer.AddActor(ballActor)
                ballActor_transparent = self.creator.CreateSphere(agent_origin, 2, color=color, opacity=.3)
                self.renderer.AddActor(ballActor_transparent)

                arrowActor = self.creator.CreateArrow(angle=direction, position=agent_origin)
                self.renderer.AddActor(arrowActor)

                range_finders = self.agent_params['rangefinders']
                lineActor_cache = []
                if range_finders:
                    for finder in range_finders:
                        lineActor = self.creator.CreateLine(finder[0], finder[1])
                        lineActor_cache.append(lineActor)
                        self.renderer.AddActor(lineActor)
                iren.GetRenderWindow().Render()

                # self.renderer.RemoveActor(arrowActor)
                self.renderer.RemoveActor(ballActor)
                # for line in lineActor_cache:
                    # self.renderer.RemoveActor(line)
            else:
                pass


def func(slope, x):
    y = slope * x ** 2
    return y


def Smoother(position_pre, position_now, orient_pre, orient_now, min_distance=2.0, nums=None):
    distance = np.linalg.norm(position_now - position_pre)
    if nums is None:
        nums = (np.ceil(distance / min_distance)).astype(np.int32)
    distance_interval = (position_now - position_pre) / nums
    orient_interval = np.where(orient_now - orient_pre < -180, (orient_now - orient_pre + 360) / nums,
                               (orient_now - orient_pre) / nums)
    positions = []
    orients = []
    for i in range(nums):
        positions.append(position_pre + i * distance_interval)
        orients.append(np.mod(orient_pre + i * orient_interval, 360))

    # print("positions", positions)
    # print("orients", orients)
    return positions, orients


def counter_clockwise_rotate(data, theta):
    mat = np.mat([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    data = np.expand_dims(np.array(data), 1)
    returned = np.array(np.matmul(mat, data))[:, 0]
    # print("returned", returned)
    return returned


def Smoother_soft(position_pre, position_now, orient_pre, orient_now, orient_pre_cam, orient_now_cam, divisions=None, min_distance=1.0):
    if divisions is None:
        distance = np.linalg.norm(position_now - position_pre)
        divisions = (np.ceil(distance / min_distance)).astype(np.int32)

    orient_interval_cam = (orient_now_cam - orient_pre_cam) / divisions
    orients_cam = []
    for i in range(divisions):
        orients_cam.append(orient_pre_cam + i * orient_interval_cam)

    orients = []
    positions = []
    if position_pre is not None:
        orient_interval = (orient_now - orient_pre) / divisions
        for i in range(divisions):
            orients.append(orient_pre + i * orient_interval)

        rotate_theta = orient_pre - np.pi / 2
        position_now_tmp = counter_clockwise_rotate((position_now - position_pre)[0:2], rotate_theta)
        coefficient = position_now_tmp[1] / (position_now_tmp[0] ** 2 + 1e-5)
        # print("coef", position_now_tmp, coefficient)
        deltas_position = position_now_tmp[0] / divisions * np.array([i+1 for i in range(divisions)])

        for i in range(len(deltas_position)):
            x = deltas_position[i]
            y = coefficient * x ** 2
            pos_tmp = counter_clockwise_rotate([x, y], - rotate_theta)
            pos_tmp = np.concatenate((pos_tmp, np.array([0.0])), 0) + position_pre
            positions.append(pos_tmp)

    return positions, orients, orients_cam


def Smoother_camera():
    orients = []
    return orients

def add_arguments(parser):
    parser.add_argument('--use_sprase_reward', default=False, help="whether use sparse reward", action='store_true')
    parser.add_argument('--use_curriculum_learning', default=False, help="whether use curriculum learning", action='store_true')
    return parser