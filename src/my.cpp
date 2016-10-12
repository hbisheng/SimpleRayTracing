#include <math.h>  
#include <cstdlib> 
#include <time.h>
#include <cstdio>  
#include <iostream>

#include "opencv2/core/core.hpp"  
#include "opencv2/features2d/features2d.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/nonfree/nonfree.hpp"  
#include "cv.h"
using namespace std;

double M_pi = 3.1415926535;
double M_1_Pi = 1.0 / M_pi;

const int w=1366*3, h=768*3 , samps = 9; // #screen size

IplImage *text1=cvLoadImage("Default_Bump.jpg",1);
IplImage *text2=cvLoadImage("six_texture.jpg",1);
CvScalar s;


class Vec {    
public:   
	double x, y, z;    // position, also color (r,g,b)
	
	Vec operator+(const Vec &b) const{ return Vec(x+b.x,y+b.y,z+b.z); }
	Vec operator-(const Vec &b) const{ return Vec(x-b.x,y-b.y,z-b.z); }
	Vec operator*(double b)const { return Vec(x*b,y*b,z*b); }
	// cross product
	Vec operator%(const Vec&b) const{return Vec(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);}

	Vec(double x_=0, double y_=0, double z_=0){ x=x_; y=y_; z=z_; }
	Vec mult(const Vec &b) { return Vec(x*b.x,y*b.y,z*b.z); }
	Vec& norm(){ return *this = *this * (1/sqrt(x*x+y*y+z*z)); }
	
	//dot product
	double dot(const Vec &b) const { return x*b.x+y*b.y+z*b.z; }
	double len(){ return sqrt(x*x+y*y+z*z); }	
};


class Ray 
{ 
public:
	Vec o;// position
	Vec d;// direction 
	Ray(Vec o_ = Vec(), Vec d_ = Vec()) : o(o_), d(d_) {} 
};

class Camera
{
public:
	Vec focus;//focus point
	Vec dir;//gaze direction
	
	Vec dy;
	Vec dz;//to get the screen
	
	Camera(Vec o_, Vec d_) : focus(o_), dir(d_) // to determine dx and dy
	{
		dy.x = 0.0; dy.y = 0.16 / w; dy.z = 0.0;
		dz.x = 0.0; dz.y = 0;  dz.z = 0.09 / h;  
	}
	
	Ray pixel_dir(int pos_w, int pos_h)//return the direction of certain pixels   d need to be norm();
	{
		Vec d = dir*0.1 + dy * (pos_w - w / 2) + dz*(pos_h - h / 2);
		Vec f = focus;
		Ray r(f,d);
		return r;
	}
};


double clamp(double x)
{ 
	return x<0 ? 0 : x>1 ? 1 : x; 
}
int toInt(double x)
{ 
	//return int(pow(clamp(x),1/2.2)*255+.5); 
	return (x > 1 ? 255 : (x < 0 ? 0 : int(255*x)));
}


enum Refl_t { DIFF, SPEC, REFR, TEXT1 ,TEXT2};  // material types, used in radiance()
enum Shape{PLANE, SPHERE, CUBE, TETRA};

class Basic_obj
{
public:
	Vec pos;
	Vec color;
	Refl_t refl;	
	Shape shape;
	virtual double intersect(const Ray &ray) const
	{
		return 0;
	}
	virtual Vec Normal_V(const Vec &intersect_p) const
	{
		return Vec(0,0,0);
	}
};

class Sphere:public Basic_obj 
{
public:
	double rad;       // radius
	Sphere(double rad_, Vec p_, Vec c_, Refl_t refl_):rad(rad_)
	{
		pos = p_;  
		color = c_; 
		refl = refl_; 
		shape = SPHERE;
	}
	double intersect(const Ray &ray)  const
	{ // returns distance, 0 if nohit
		Vec op = ray.o - pos; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
		double t, eps = 1e-4;
		double b = op.dot(ray.d);
		double det = b*b - op.dot(op)+rad*rad;
		int num = 0;		
		//cerr << "I'm alive" << endl;
		if (det<0) 
			return 0; 
		else 
			det=sqrt(det);
		return (t=-b-det)>eps ? t : ((t=-b+det)>eps ? t : 0); 
	}
	Vec Normal_V(const Vec& intersect_p) const
	{
		Vec N = (intersect_p-pos).norm(); 
		return N;
	}
};

class Plane:public Basic_obj
{
public:
	Vec N; // Normal Vector
	Plane(Vec N_, Vec p_, Vec c_, Refl_t refl_): N(N_)
	{
		pos = p_;
		color = c_;
		refl = refl_;
		shape = PLANE;
	}
	double intersect(const Ray& ray) const
	{
		double t, eps = 1e-4;
		Vec n = N;
		Vec p = pos;
		if(n.dot(ray.d) == 0)
			return 0;
		else
			t = (n.dot(p) - n.dot(ray.o)) / (n.dot(ray.d));
		return (t > eps? t : 0);
	}
	Vec Normal_V(const Vec& intersect_p) const
	{
		Vec n = N;
		//cerr << n.x << ' ' << n.y << ' ' << n.z << endl;
		return n;
	}
};

class Cube:public Basic_obj
{
public:
	Ray LWH;
	Ray side[6];
	Cube(Ray LWH_, Vec p_, Vec c_, Refl_t refl_): LWH(LWH_)
	{
		pos = p_;
		color = c_;
		refl = refl_;
		shape = CUBE;
		
		//generate six related plane
		//front back
		side[0].o = pos + LWH.d * (LWH.o.x / 2.); 
		side[0].d = LWH.d;
		
		side[1].o = pos - LWH.d * (LWH.o.x / 2.); 
		side[1].d = LWH.d*-1;
		
		//up down
		side[2].o = pos + Vec(0,0,1) * (LWH.o.z / 2.); 
		side[2].d = Vec(0,0,1);
		
		side[3].o = pos - Vec(0,0,1) * (LWH.o.z / 2.); 
		side[3].d = Vec(0,0,-1);
		
		//left right
		side[4].o = pos + Vec(LWH.d.y*-1.0, LWH.d.x*+1.0) * (LWH.o.y / 2.0); 
		side[4].d = Vec(LWH.d.y*-1.0, LWH.d.x*1.0,0);
		
		side[5].o = pos - Vec(LWH.d.y*-1.0, LWH.d.x*+1.0) * (LWH.o.y / 2.0); 
		side[5].d = Vec(LWH.d.y*1.0, LWH.d.x*-1.0,0);
	}
	double intersect(const Ray& ray) const
	{
		double t[6], t_max[3], t_min[3], eps = 1e-4;
		for(int i = 0; i != 6; i+= 1)
		{
			Vec n = side[i].d, p = side[i].o;
			if(n.dot(ray.d) == 0)
				t[i] =  0;
			else
				t[i] = (n.dot(p) - n.dot(ray.o)) / (n.dot(ray.d));
		}
		for(int i = 0; i != 3; i++)
		{
			t_max[i] = max(t[2*i],t[2*i+1]);
			t_min[i] = min(t[2*i],t[2*i+1]);
			if(t[2*i] < 0 & t[2*i +1] < 0)
			
				return 0;
		}
		double ttmax = min(min(t_max[0],t_max[1]),t_max[2]), ttmin = max(max(t_min[0],t_min[1]),t_min[2]);
		
		
		return (ttmax - ttmin > eps? ((ttmin > eps)? ttmin: 0) : 0);

		
	}
	Vec Normal_V(const Vec& intersect_p) const
	{
		Vec x = intersect_p;
		for(int i = 0; i != 6; i++)
		{
			Vec n = side[i].d;
			Vec p = side[i].o;
			if( x.dot(n) == p.dot(n) )
				return n;
		}
	}
};

class Tetrahedron: public Basic_obj
{
	Vec Triangle(int which, const Ray& ray) const
	{
		//the 1st triangle use p234
		Vec point[3];
		int count = 0;
		for(int i = 0 ; i != 4; i++)
			if(i != which)
				point[count++] = p[i];
		assert(count == 3);
		Vec S = point[0] - ray.o;
		Vec E1 = point[0] - point[1];
		Vec E2 = point[0] - point[2];
		
		return ( Vec( Det(S, E1, E2), 
					  Det(ray.d, S, E2), 
					  Det(ray.d, E1, S)) * (1.0 / Det(ray.d, E1 , E2) )  );
	}
	
	double Det(Vec a, Vec b, Vec c) const
	{
		return( a.x*b.y*c.z + b.x*c.y*a.z + c.x*a.y*b.z - c.x*b.y*a.z - b.x*a.y*c.z - a.x*c.y*b.z);
	}
	
	Vec parse_normal(int which, const Vec& intersect_p) const
	{
		Vec point[3];
		int count = 0;
		for(int i = 0 ; i != 4; i++)
			if(i != which)
				point[count++] = p[i];
		assert(count == 3);		
		
		Vec n = ((point[0] - point[1]) % (point[0] - point[2])).norm();		
		
		if( abs(point[0].dot(n) - intersect_p.dot(n)) < 1e-4)
			return n;
		else
			return Vec();
	}
	
public:
	Vec p[4]; // Normal Vector
	Tetrahedron(Ray p12, Ray p34, Vec c_, Refl_t refl_)
	{
		p[0] = p12.o;
		p[1] = p12.d;
		p[2] = p34.o;
		p[3] = p34.d;
		
		this->pos = (p[0] + p[1] + p[2] + p[3] ) * 0.25;
		color = c_;
		refl = refl_;
		shape = TETRA;
	}
	double intersect(const Ray& ray) const
	{
		double t[4], beta[4], gama[4], eps = 1e-4;
		double small_p = 1e20;
		for(int i = 0; i != 4; i++)
		{			
			Vec ans = Triangle(i, ray);	
			t[i] = ans.x;
			beta[i] = ans.y;
			gama[i] = ans.z;
			if(beta[i] >= 0 && beta[i] <= 1)	
				if(gama[i] >= 0 && gama[i] <=  1)
					if(beta[i]+gama[i]<=1)
						if(t[i] >= eps && t[i] < small_p)						
							small_p = t[i];	
		}	
		return ((small_p < 1e20) ? small_p : 0);	
	}
	Vec Normal_V(const Vec& intersect_p) const
	{
		Vec ans;
		Vec rt;
		for(int i = 0 ; i != 4; i++)	
		{	
			ans = parse_normal(i, intersect_p);
			if(ans.x != 0 || ans.y != 0 || ans.z != 0)
				rt = ans;
		}
		assert(rt.x != 0 || rt.y != 0 || rt.z != 0);
		return rt;
	}	
};


const int Up = 30, Right = 30, Front = 30;
const int Down = -30, Left = -30, Back = -30; 

Vec Lights[] =
{
	Vec (30.5,-0.5,9),Vec(30.25,-0.5,9), Vec (30,-0.5,9),Vec(29.75, -0.5, 9),Vec (29.5,-0.5,9),
	Vec (30.5,-0.25,9),Vec(30.25,-0.25,9), Vec (30,-0.25,9),Vec(29.75, -0.25, 9),Vec (29.5,-0.25,9),
	Vec (30.5,0,9), Vec(30.25,0,9),Vec (30,0,9),Vec(29.75, 0, 9),Vec (29.5,0,9),
	Vec (30.5,0.25,9),Vec(30.25,0.25,9), Vec (30,0.25,9),Vec(29.75, 0.25, 9),Vec (29.5,0.25,9),
	Vec (30.5,0.5,9), Vec(30.25,0.5,9),Vec (30,0.5,9),Vec(29.75, 0.5, 9), Vec (29.5,0.5,9)
};



int things_num = 11;
Basic_obj* things[] = 
{
	new Sphere(3.6, Vec(28, 4, -6.4), Vec(0.75,0.75,0.25), TEXT1),
	//new Sphere(3.2, Vec(25, 10, -3.8), Vec(0.75,0.75,0.25), TEXT1),
	new Plane(Vec(0,1,0), Vec(0,-15,0), Vec(0.88,0.53,0.53), DIFF), //Left 
	new Plane(Vec(0,1,0), Vec(0,15,0),Vec(0.53,0.53,0.88), DIFF), // right
	new Plane(Vec(1,0,0), Vec(40,0,0),Vec(0.88,0.88,0.88), SPEC), // back
	new Plane(Vec(0,0,1), Vec(0,0,10),Vec(0.88,0.88,0.88), DIFF), // up
	new Plane(Vec(0,0,-1), Vec(0,0,-10),Vec(0.88,0.88,0.88), TEXT2),// down
	new Plane(Vec(1,0,0), Vec(-100,0,0),Vec(0,0,0), DIFF), // front
	new Sphere(5.5, Vec(30, 0, 15), Vec(30,30,30), DIFF),// the light
	new Sphere(3.5,Vec(29, -8, -6.5), Vec(0.99,0.99,0.99),REFR), // the REFR one
	new Cube(Ray(Vec(3,3,3),Vec(1,0,0).norm()),Vec(25,+10,-8.5),Vec(0.53,0.88,0.53), DIFF),
	//new Tetrahedron(Ray(Vec(31, -2, -10),Vec(23, -3.2, -10)),Ray(Vec(27,-4.2,-6.5), Vec(26.5, -6.5, -10)), Vec(.45,.15,.75), DIFF) 
	new Tetrahedron(Ray(Vec(26, -1, -10),Vec(18, -2.2, -10)),Ray(Vec(22,-3.2,-5), Vec(21.5, -5.5, -10)), Vec(0.88,0.53,0.88), DIFF) 
};




bool intersect(const Ray &ray, double &t, int &id)
{
	double d;
	double inf=t=1e20;
	for(int i = 0; i < things_num; i++) 
		if((d= things[i]->intersect(ray)) && d<t)
		{	
			t=d;
			id=i;
		}
	return t<inf;
}

Vec radiance(const Ray &ray){
	//cout << "into radiance" << endl;
	double distance;                               // distance to intersection                         // id of intersected object
	int id = 0;
	if (!intersect(ray, distance, id)) 
	{
		cout << "not intersect" << endl;
		return Vec(0,0,0); // if miss, return black
	}
	const Basic_obj* obj = things[id];        // the hit object
	
	Vec x = ray.o+ray.d*distance;//intercect point
	Vec V = Vec() - ray.d;
	Vec N = obj->Normal_V(x);
	
	if(V.dot(N) < 0)
		N = Vec() - N;
		
	Vec f = obj->color;
	
	//if(obj->shape == TETRA)
	//	cout << "Tetra N" << N.x << ' ' << N.y << ' ' << N.z << endl;
	
	//cout << "after color" << endl;	
	//cerr <<  (obj->refl == DIFF) << endl;	
	if(obj->refl == DIFF || obj-> refl == TEXT1 || obj-> refl == TEXT2) // Ideal DIFFUSE reflection
	{
		//Phong model
		double Ii = 1;
		Vec Kd;
		if(obj->refl == DIFF)
			Kd = obj->color;
		else
		{
			double XX, YY;
			if(obj->shape == SPHERE)
			{
				Vec xo = (x - obj->pos);
				Vec No; No.x = xo.x; No.y=xo.y; No.z = 0; No = No.norm();
				XX = acos( No.dot(Vec(1,0,0))) *M_1_Pi / 0.75;
				YY = asin( xo.z / xo.len()) * M_1_Pi / 0.75;
			}
			else if(obj->shape == PLANE)
			{
				XX = x.x/3; // /how many meters an image
				YY = x.y/3;
			}
			
			int SCALE = 1;
			int i = int ( 1.0*(XX - SCALE * (int(XX) / SCALE)) / SCALE *text1->height);
			int j = int ( 1.0*(YY - SCALE * (int(YY) / SCALE)) / SCALE *text1->width);
			
			if(obj->refl == TEXT1)
			{
				while(i < 0)
					i += text1->height;
				while(j < 0)
					j += text1->width;
				s=cvGet2D(text1,i,j); // get the (i,j) pixel value
			
			}
			
			else if(obj->refl == TEXT2)
			{
				while(i < 0)
					i += text2->height;
				while(j < 0)
					j += text2->width;
				s=cvGet2D(text2,i,j); // get the (i,j) pixel value
			}
			
			Kd.x = (double)s.val[0]/200.0 ;
			Kd.y = (double)s.val[1]/200.0 ;
			Kd.z = (double)s.val[2]/200.0 ;
		}
		Vec Ks = Kd * 0.2;
		Vec Ka = Kd * 0.43;
		//cout << L.x << ' '<< L.y << ' ' << L.z << endl;
		//cout << N.x << ' '<< N.y << ' ' << N.z << endl;
		//cout << L.dot(N) << endl;
		
		int Light_num=sizeof(Lights)/sizeof(Vec);
		Vec total;
		for(int i = 0; i != Light_num; i++)
		{
			Vec t1,t2,t3;
			Vec L = (Lights[i] - x).norm();
			Vec R;
			if(L.dot(N) > 1e-4)
			{
				t1 = Kd*Ii*(pow(L.dot(N),2));
				R = (N*2*(N.dot(L)) - L).norm();//否则R为0，不考虑下边的高光
			}
			if(R.dot(V) > 1e-4)
				t2 = Ks*Ii*(pow(R.dot(V),20));
			t3 = Ka* Ii;
			//shadows the part that is shadowed by other object
			double d;
			int id2;
			bool c = intersect( Ray(x, L), d, id2); 
			if(d < (Lights[i] - x).len() )
			{
		
				t1 = Vec();
				t2 = Vec();
			}
			total = total + (t1 + t2 + t3) * (1.0 / Light_num);
		}
		return total; 
	}
	else if(obj->refl == SPEC)//Ideal SPECULAR reflection
	{	
		Vec R_ofV = (N*2*(N.dot(V)) - V).norm();
		return f.mult(radiance(Ray(x, R_ofV)));
	}
	else if(obj->refl == REFR)//Ideal dielectric REFRACTION
	{	
		Vec n = obj->Normal_V(x);
		Vec nl = (N.dot(ray.d) < 0) ? N:(Vec()-N);
		bool into = N.dot(nl)>0;                // Ray from outside going in?
		double nnt = (into?1.0/1.5:1.5);
		double ddn= nl.dot(ray.d);
		double cos2t;
		Ray reflRay(x, ray.d-n*2*n.dot(ray.d));
		
		if ((cos2t=1-nnt*nnt*(1-ddn*ddn))<0)    // Total internal reflection
			return f.mult(radiance(reflRay));
				
		Vec tdir = (ray.d*nnt - n*((into?1:-1)*(ddn*nnt+sqrt(cos2t)))).norm();
		
		double a=1.5 - 1, b=1.5 + 1; 
		double R0=a*a/(b*b), c = 1-(into?-ddn:tdir.dot(n));
		double Re=R0+(1-R0)*c*c*c*c*c,Tr=1-Re,P=.25+.5*Re,RP=Re/P,TP=Tr/(1-P);
		
		//if((double)rand() / (double)RAND_MAX > 0.2)
		//	return f.mult(radiance(reflRay)*Re+radiance(Ray(x,tdir))*Tr);
		//else
			return f.mult(radiance(Ray(x,tdir)));
	}
}


int main(){	
	freopen("ans.txt","w",stdout);
	cout << "start" << endl;
	Camera cam(Vec(-5,0,0),Vec(1,0,0));
	//w=1366, h=768
	unsigned srand(time(NULL));
	
	
	Vec** screen;
	screen = new Vec*[w];
	for(int i = 0; i != w; i++)
		screen[i] = new Vec[h];
	Vec r;
	for (int y=0; y < h; y++)
	{	
		fprintf(stderr,"\rRendering %5.2f%%", 100.* y /(h - 1));
		for (int x=0; x < w; x++)   // Loop cols
		{	
			int n = sqrt(samps);
			for (int zz = 0; zz != n; zz++)
				for(int yy = 0; yy != n; yy++)
				{
					r = Vec();	

					Vec dir = (cam.pixel_dir(x,y).d + cam.dy * (-0.5+0.5/ n + yy / n) + cam.dz * (-0.5+0.5/ n + zz / n)).norm();
					r = r + radiance(Ray(cam.focus, dir));
					screen[x][y] = screen[x][y]+ r * (1.0 / samps);	
				} 
		}
	}
	
	cout << "draw_pixel done" << endl;
	
	Ray ray(Vec(0,0,0),Vec(1,0,0));
	double distance; int id;
	bool a = intersect(ray, distance, id);
	
	
	FILE *f = fopen("cube.ppm", "w");         // Write image to PPM file.
	fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);	
	for(int y = h - 1; y >= 0; y--)
		for(int x = 0; x < w; x++)
			fprintf(f,"%d %d %d ", toInt(screen[x][y].x), toInt(screen[x][y].y), toInt(screen[x][y].z));	
	
	return 0;
}

