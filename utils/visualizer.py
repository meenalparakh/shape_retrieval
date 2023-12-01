import meshcat
import meshcat.geometry as g

# from pyngrok import ngrok


class VizServer:
    def __init__(self, port_vis=6000) -> None:
        zmq_url = f"tcp://127.0.0.1:{port_vis}"
        self.mc_vis = meshcat.Visualizer(zmq_url=zmq_url).open()
        self.mc_vis["scene"].delete()
        self.mc_vis["meshcat"].delete()
        self.mc_vis["/"].delete()

    def view_pcd(self, pts, colors=None, name="scene", size=2.0):
        if colors is None:
            colors = pts
        # self.mc_vis["scene"].delete()
        self.mc_vis["scene/" + name].set_object(
            g.PointCloud(pts.T, color=colors.T / 255.0, size=size)
        )

    def close(self):
        self.mc_vis.close()
