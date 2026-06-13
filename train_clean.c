/*This code is logic-heavy and has been conveniently divided into sections by the author. The author wants to make it known that these comments are aimed at providing an overview of what each section is doing. He also acknowledges that he should have made separate source files for ease of debugging but admits that he is unfortunately too lazy to do that.*/
/*Section 1: Headers. Macros, definitions, shared memory. Read through these carefully, most of the logic is actually contained here.*/
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdint.h>
/* ML Macros */
#define K 8 // Pretty good for sine function I guess
#define N 100 //Training data; I learnt from Andrew Ng so this is just N
#define EPOCH 5000
#define ALPHA 1e-6//Minimum value of learning rate, alpha
#define NOISE 0.05 //Gaussian noise on sine
/* Macros for our scheduler and patcher; Santa's got a macro for everyone */
#define SCHED_INTERVAL 50
#define SCHED_HISTORY 8
#define SENTINEL_A 0xDEADBEEFCAFEBABEULL//dead beef cafe babe!!
#define SENTINEL_B 0xFEEDFACEDEADC0DEULL //feed face dead code: random enough to not be instructions

#ifdef DEBUG
    #define DBG_PRINTF(fmt, ...) do { printf(fmt, ##__VA_ARGS__); fflush(stdout); } while (0)
#else
    #define DBG_PRINTF(fmt, ...) do { } while (0)
#endif

/* The shared memory */
typedef struct{
	double weights[K+1];
	double grads[K+1];
	double loss;
	int iter;
	int scheduler_flag;
	int patching;
	double loss_history[8];
	int history_idx;
	//The following members facilitate the children in delegating the patching process to their parent.
	double scheduled_alpha;
	int apply_patch;
} Shared;
/* Section 2. Data loader. Nothing to see here, just basic IO. Keep in mind that xs and ys are both static because we love our children, don't we? */
static double xs[N];
static double ys[N];
static void load(){
	FILE *f=fopen("data.bin","rb");
	if(!f){
		perror("fopen data.bin");
		exit(1);
	}
	fread(xs,sizeof(double),N,f);
	fread(ys,sizeof(double),N,f);
	fclose(f);
	//Feature scaling [-PI,PI]->[-1,1]
	for(int i=0;i<N;i++)
		xs[i]/=M_PI;
}
/* Section 3. Tighten your seat belts people, this is an absolutely criminal way of input-process-output and must not be repeated. Because alpha is so holy, it lives in .text, where we put it, find it and patch it. */
__attribute__((noinline)) static double get_alpha(){ //gdb helped discover this issue; we don't want this to go inline at all
	double alpha;
	__asm__ volatile (
        "jmp 1f\n\t"
        ".quad 0xDEADBEEFCAFEBABE\n\t"
        "1:\n\t"
        "movabsq $0x3F1A36E2EB1C432d, %%rax\n\t"
        "movq %%rax, %0\n\t"
        "jmp 2f\n\t"
        ".quad 0xFEEDFACEDEADC0DE\n\t"
        "2:\n\t"
        : "=x"(alpha)
        :
        : "rax", "memory"
    );	return alpha;
}
static void get_bounds(void **start, void **end) {
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) { perror("fopen /proc/self/maps"); exit(1); }

    uintptr_t target = (uintptr_t)get_alpha;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "r-xp") == NULL && strstr(line,"rwxp")==NULL) continue;

        unsigned long s, e;
        sscanf(line, "%lx-%lx", &s, &e);

        if (target >= s && target < e) {
            *start = (void *)s;
            *end   = (void *)e;
            fclose(f);
            return;
        }
    }

    fprintf(stderr, "could not find .text segment\n");
    exit(1);
}//Presenting my absolutely-not-needed patcher below. Hope you enjoy, I am not gonna speak much in the patcher!
static void patch(Shared *sh,double new_alpha){
	void *text_start, *text_end; 
	get_bounds(&text_start,&text_end);
	uint8_t *p = (uint8_t *)text_start;
        uint8_t *end = (uint8_t *)text_end;
	DBG_PRINTF("scanning .text: %p to %p\n", text_start, text_end);
	DBG_PRINTF("get_alpha is at: %p\n", (void*)get_alpha);
        uint64_t sa = SENTINEL_A;
        uint64_t sb = SENTINEL_B;
	while(p+8+2+8+8<end){
		uint8_t *sa_addr=p+2;
		if(memcmp(sa_addr,&sa,8)!=0){
			p++;
			continue;
		}
		uint8_t *imm_addr=sa_addr+8+2;
		//Debug print
		DBG_PRINTF("found SENTINEL_A at %p\n", (void*)p);
		for(int i = 0; i < 32; i++)
			DBG_PRINTF("%02x ", p[i]);
		DBG_PRINTF("\n");
		if (memcmp(p+27, &sb, 8) != 0) {  
			p++;
		       	continue;
	       	}
		/* I FOUND IT MAMMA */
		long size=sysconf(_SC_PAGESIZE); //100th line. Look at that, Tendulkar!
		void* page=(void *)((uintptr_t)imm_addr& ~(size-1));
		sh->patching=1;
		if(mprotect(page,size,PROT_READ|PROT_WRITE|PROT_EXEC)!=0){
			perror("mprotect RWX");
			exit(1);
		}
		memcpy(imm_addr,&new_alpha,sizeof(double)); //patching
		
		/* Evict stale CPU cache entries to synchronize D-Cache modifications into the I-Cache */
		__builtin___clear_cache((char *)imm_addr, (char *)imm_addr + sizeof(double));

		if(mprotect(page,size,PROT_READ|PROT_EXEC)!=0){
			perror("mprotect RW");
			exit(1);
		}
		sh->patching=0;
		printf("[scheduler] patched alpha → %.8f\n", new_alpha);
        	return;
    }
    printf("sentinel pattern not found in .text\n");
    exit(1);
}
/* Section 4. Worker children. Each worker takes one weight and calculates the partial derivative for that weight, then writesit to Shared->grads. Worker 0 has an extra job of keeping the losses just because someone has to. */
static void worker(Shared *sh,int k){
	double grad=0.0;
	double loss=0.0;
	for(int i=0;i<N;i++){
		double y_hat=0.0; /*If you are reading in a version where I have uploaded my formulae refer to them; else wait*/	      double x_pow=1.0;
		for(int j=0;j<=K;j++){
			y_hat+=sh->weights[j]*x_pow;
			x_pow*=xs[i];
		}
		double res=y_hat-ys[i]; //residual
		double xk=pow(xs[i],k);
		grad+=res*xk;
		if(k==0)
			loss+=res*res;
	}
		grad*=2.0/N;
		sh->grads[k]=grad;
		if(k==0)
			sh->loss=loss/N;
		_exit(0);
}
/* Section 5. Scheduler child. */
static void sched(Shared *sh){
	double current_alpha=0.0001;
	while(1){
		while(sh->scheduler_flag==0)
			usleep(1000); //sleep till main signals
		sh->scheduler_flag=0;
		sh->loss_history[sh->history_idx]=sh->loss;
		sh->history_idx=(sh->history_idx+1)%SCHED_HISTORY;
		if (sh->iter< SCHED_INTERVAL * SCHED_HISTORY)
		       	continue;
		/* In full window now. Find min and max in window. */
		double min_loss=sh->loss_history[0];
		double max_loss=sh->loss_history[0];
		for(int i=1;i<SCHED_HISTORY;i++){
			if(sh->loss_history[i]<min_loss)
				min_loss=sh->loss_history[i];
			if(sh->loss_history[i]>max_loss)
				max_loss=sh->loss_history[i];
		}
		/* Plateau detection at <1% improvement */
		double improvement=(max_loss-min_loss)/(max_loss+1e-12);
		if(improvement<0.02){
			current_alpha*=0.5;
			if(current_alpha<ALPHA)
				current_alpha=ALPHA;
			sh->scheduled_alpha=current_alpha;
			sh->apply_patch=1;
			while(sh->apply_patch)
				usleep(1000);
		}
	}
	fprintf(stderr, "scheduler exited unexpectedly\n");
	exit(1);
}
/* Section 6. main. main also has several sections, keep reading the comments to get a complete overview. You're doing great! Just a few more eccentricities to witness. */
int main(){
	/* Part 1. Startup work. */
	load();
	//Allocate shared memory
	Shared *sh=mmap(NULL,sizeof(Shared),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);
	if(sh==MAP_FAILED)
	{
		perror("mmap");
		return 1;
	}
	//Initialize weights
	srand(99);
	for (int k = 0; k <= K; k++)
	        sh->weights[k] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
	//memset shared memory
	memset(sh->grads, 0, sizeof(sh->grads));
   	memset(sh->loss_history, 0, sizeof(sh->loss_history));
    	sh->loss = 0.0;
    	sh->iter= 0;
    	sh->scheduler_flag = 0;
    	sh->history_idx = 0;
	//fork scheduler
	pid_t sched_pid=fork();
	if(sched_pid<0)
	{
		perror("fork scheduler");
		exit(1);
	}
	if(sched_pid==0)
		sched(sh); //child never returns
	//I am adding a debug-only message to confirm that the patcher is running fine. The next two lines may be commented out.
	//patch(sh,0.01);
	//printf("Patcher OK\n");
	DBG_PRINTF("forking workers...\n");
	/*Part 2. Training loop.*/
	for (int iter = 0; iter <EPOCH; iter++) {
        	sh->iter = iter;
		if(sh->apply_patch){
			patch(sh,sh->scheduled_alpha);
			sh->apply_patch=0;
		}
        	/* fork one worker per weight */
        	pid_t pids[K+1];
        	for (int k = 0; k <= K; k++) {
            		pids[k] = fork();
            		if (pids[k] < 0) { perror("fork worker"); return 1; }
            		if (pids[k] == 0){DBG_PRINTF("worker %d started\n",k); worker(sh, k);}  /* child never returns */
      DBG_PRINTF("forked worker %d pid=%d\n", k, pids[k]);
		}
		
	        /* wait for all workers */
        	for (int k = 0; k <= K; k++)
            	waitpid(pids[k], NULL, 0);
	        while(sh->patching)
			usleep(100); //prevent race condition
       		 /* read alpha from .text and update weights */
		DBG_PRINTF("calling get_alpha\n");
		
		__asm__ volatile("" ::: "memory"); //compiler memory barrier to prevent LICM
        	double alpha = get_alpha();
		
		DBG_PRINTF("got alpha: %f\n", alpha);
        	for (int k = 0; k <= K; k++)
            		sh->weights[k] -= alpha * sh->grads[k];
	
        	/* signal scheduler every SCHED_INTERVAL iterations */
        	if (iter % SCHED_INTERVAL == 0) {
            		sh->scheduler_flag = 1;
            		printf("[iter %4d] loss=%.6f alpha=%.8f\n", iter, sh->loss, alpha);
        	}
    	}
	/* Part 3. Cleanup */
	kill(sched_pid,SIGTERM);
	waitpid(sched_pid,NULL,0);
	printf("\n FINAL WEIGHTS: \n");
	for(int k=0;k<K;k++)
		printf("	w%d= %+.6f\n",k,sh->weights[k]);
	printf("\n FINAL LOSS: %.6f\n",sh->loss);
	munmap(sh,sizeof(Shared));
	return 0;
}
/* End of program. Proudly hand-typed in vim by Aadity Setu (@lord-of-the-strings), all rights reserved as per MIT licence. */
